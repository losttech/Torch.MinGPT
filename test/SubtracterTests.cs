namespace ClassLib;

public class SubtracterTests {
    [Fact]
    public void AddsCorrectly() {
        int diff = new Subtracter().Subtract(20, 22);

        Assert.Equal(-2, diff);
    }
}
