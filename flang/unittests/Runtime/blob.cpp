// Basic sanity tests of blobs
#include "testing.h"
#include "../../runtime/blob-builder.h"
#include <cstring>

using namespace Fortran::runtime;

static void PODblob() {
  struct POD {
    int j, k;
  };
  POD pod{2, 3};
  Terminator terminator{__FILE__, __LINE__};
  OwningPtr<TypedBlob<POD>> blob{CreateBlob(terminator, pod)};
  if (blob->Get().j != 2) {
    Fail() << "POD j " << blob->Get().j << " should be 2\n";
  }
  if (blob->Get().k != 3) {
    Fail() << "POD k " << blob->Get().k << " should be 3\n";
  }
}

static void tupleBlob() {
  struct POD {
    int j, k;
  };
  using Tuple = std::tuple<POD, const char *, std::tuple<int, int>>;
  Tuple x;
  std::get<0>(x).j = 4;
  std::get<0>(x).k = 5;
  std::get<1>(x) = "Hi!";
  std::get<2>(x) = std::tuple<int, int>{6, 7};
  Terminator terminator{__FILE__, __LINE__};
  OwningPtr<TypedBlob<Tuple>> blob{CreateBlob(terminator, x)};
  int n;
  if ((n = blob->get<0>().j) != 4) {
    Fail() << "tuple get j " << n << " should be 4\n";
  }
  if ((n = blob->get<0>().k) != 5) {
    Fail() << "tuple get k " << n << " should be 5\n";
  }
  const char *s;
  if (std::strcmp(s = blob->get<1>(), "Hi!") != 0) {
    Fail() << "tuple str '" << s << "' should be 'Hi!'\n";
  }
  std::tuple<const int &, const int &> intTup{blob->get<2>()};
  if ((n = std::get<0>(intTup)) != 6) {
    Fail() << "intTup get<0> " << n << " should be 6\n";
  }
  if ((n = std::get<1>(intTup)) != 7) {
    Fail() << "intTup get<1> " << n << " should be 7\n";
  }
  if ((n = std::get<0>(blob->get<2>())) != 6) {
    Fail() << "tuple nested get<0> " << n << " should be 6\n";
  }
  if ((n = std::get<1>(blob->get<2>())) != 7) {
    Fail() << "tuple nested get<1> " << n << " should be 7\n";
  }
  if ((n = std::get<0>(blob->Get()).j) != 4) {
    Fail() << "tuple Get().get j " << n << " should be 4\n";
  }
  if ((n = std::get<0>(blob->Get()).k) != 5) {
    Fail() << "tuple Get().get k " << n << " should be 5\n";
  }
  if (std::strcmp(s = std::get<1>(blob->Get()), "Hi!") != 0) {
    Fail() << "tuple Get() str '" << s << "' should be 'Hi!'\n";
  }
}

static void vectorBlob() {
  using Vector = std::vector<const char *>;
  Vector x{"a", "bc", "def", nullptr, ""};
  Terminator terminator{__FILE__, __LINE__};
  OwningPtr<TypedBlob<Vector>> blob{CreateBlob(terminator, x)};
  if (blob->size() != 5) {
    Fail() << "vector size " << blob->size() << " should be 5\n";
  }
  const char *ptrs[]{"a", "bc", "def", nullptr, ""};
  std::size_t j{0};
  for (const char *str : *blob) {
    if (j > 4) {
      Fail() << "vector has too many elements\n";
      break;
    }
    if (str) {
      if (!ptrs[j]) {
        Fail() << "vector [" << j << "] is '" << str
               << "' but should be null\n";
      } else if (std::strcmp(str, ptrs[j]) != 0) {
        Fail() << "vector [" << j << "] is '" << str << "' but should be '"
               << ptrs[j] << "'\n";
      }
    } else if (ptrs[j]) {
      Fail() << "vector [" << j << "] is null but should be '" << ptrs[j]
             << "'\n";
    }
    ++j;
  }
  if (j != 5) {
    Fail() << "vector has too few elements (" << j << ")\n";
  }
}

static void optionalBlob() {
  using Optional = std::optional<std::tuple<const char *>>;
  Optional x;
  Terminator terminator{__FILE__, __LINE__};
  OwningPtr<TypedBlob<Optional>> blob0{CreateBlob(terminator, x)};
  if (blob0->has_value() || *blob0) {
    Fail() << "empty optional doesn't look empty\n";
  }
  x = std::tuple<const char *>{"hi"};
  OwningPtr<TypedBlob<Optional>> blob1{CreateBlob(terminator, x)};
  if (!blob1->has_value() || !(bool)*blob1) {
    Fail() << "nonempty optional looks empty\n";
  } else {
    std::tuple<const char *> tup{**blob1};
    if (std::strcmp(std::get<0>(tup), "hi") != 0) {
      Fail() << "nonempty optional contains '" << std::get<0>(tup) << "'\n";
    }
  }
}

static void uniqueBlob() {
  using Unique = std::unique_ptr<std::vector<const char *>>;
  Unique x;
  Terminator terminator{__FILE__, __LINE__};
  OwningPtr<TypedBlob<Unique>> blob0{CreateBlob(terminator, x)};
  if (*blob0) {
    Fail() << "null unique_ptr doesn't look null\n";
  }
  std::vector<const char *> vect{"hi", "there"};
  x.reset(&vect);
  OwningPtr<TypedBlob<Unique>> blob1{CreateBlob(terminator, x)};
  x.release();
  if (!*blob1) {
    Fail() << "non-null unique_ptr looks null\n";
  } else {
    if ((**blob1).size() != 2) {
      Fail() << "non-null unique_ptr vector size " << (**blob1).size()
             << ", not 2\n";
    }
    const char *p;
    if (std::strcmp((p = (**blob1).at(0)), "hi") != 0) {
      Fail() << "non-null unique_ptr contains '" << p << "', not 'hi'\n";
    }
    if (std::strcmp((p = (**blob1).at(1)), "there") != 0) {
      Fail() << "non-null unique_ptr contains '" << p << "', not 'there'\n";
    }
  }
}

int main() {
  StartTests();
  PODblob();
  tupleBlob();
  vectorBlob();
  optionalBlob();
  uniqueBlob();
  return EndTests();
}
