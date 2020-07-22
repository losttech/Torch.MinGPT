namespace ClassLib {
    using System;

    using Xunit;

    public class AdderTests {
        [Fact]
        public void AddsCorrectly() {
            int sum = new Adder().Add(20, 22);

            Assert.Equal(42, sum);
        }
    }
}
