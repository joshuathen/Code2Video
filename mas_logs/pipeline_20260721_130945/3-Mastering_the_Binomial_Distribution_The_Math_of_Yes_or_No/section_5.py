from manim import *
import math

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout with provided title and lecture lines
        self.setup_layout(
            "The Shape of the Distribution",
            [
                "A histogram visualizes how probabilities shift with n and p.",
                "Low p-values skew the distribution toward the left.",
                "As trials increase, the shape resembles a smooth bell curve."
            ]
        )

        def binomial_pmf(n, p, k):
            if k < 0 or k > n:
                return 0
            return math.comb(n, k) * (p**k) * ((1-p)**(n-k))

        def get_chart(n, p, bar_color):
            values = [binomial_pmf(n, p, k) for k in range(n + 1)]
            # Adjust labels for dense data
            if n > 15:
                bar_names = [str(k) if k % 10 == 0 else "" for k in range(n + 1)]
                y_max = 0.2
            else:
                bar_names = [str(k) for k in range(n + 1)]
                y_max = 0.5
            
            chart = BarChart(
                values=values,
                bar_names=bar_names,
                y_range=[0, y_max, y_max/5],
                x_length=5.5,
                y_length=4,
                axis_config={"font_size": 18},
                bar_colors=[bar_color] * (n + 1)
            )
            return chart

        # === Animation for Lecture Line 1 ===
        # Show a symmetrical histogram for n=10, p=0.5 in white (#FFFFFF).
        n1 = 10
        p1 = 0.5
        chart = get_chart(n1, p1, WHITE)
        # Resolved Issue 41: Positioned at C3-F6
        self.place_in_area(chart, "C3", "F6", scale_factor=0.75)
        
        params = VGroup(
            Text(f"n = {n1}", font_size=24, color=WHITE),
            Text(f"p = {p1}", font_size=24, color=WHITE)
        ).arrange(RIGHT, buff=0.8)
        # Resolved Issue 40: Positioned in area B3-B5
        self.place_in_area(params, "B3", "B5", scale_factor=0.8)

        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(FadeIn(chart), FadeIn(params))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Shift the bars left for p=0.1, coloring them (#FF9999).
        p2 = 0.1
        chart2 = get_chart(n1, p2, "#FF9999")
        self.place_in_area(chart2, "C3", "F6", scale_factor=0.75)
        
        params2 = VGroup(
            Text(f"n = {n1}", font_size=24, color="#FF9999"),
            Text(f"p = {p2}", font_size=24, color="#FF9999")
        ).arrange(RIGHT, buff=0.8)
        self.place_in_area(params2, "B3", "B5", scale_factor=0.8)

        self.play(self.lecture[1].animate.set_color("#FF9999"))
        self.play(
            Transform(chart, chart2),
            Transform(params, params2),
            run_time=2
        )
        self.play(Indicate(chart))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Morph bars into a smooth bell curve for large n=50 (#55FF55).
        n3 = 50
        p3 = 0.5
        chart3 = get_chart(n3, p3, "#55FF55")
        self.place_in_area(chart3, "C3", "F6", scale_factor=0.75)
        
        params3 = VGroup(
            Text(f"n = {n3}", font_size=24, color="#55FF55"),
            Text(f"p = {p3}", font_size=24, color="#55FF55")
        ).arrange(RIGHT, buff=0.8)
        self.place_in_area(params3, "B3", "B5", scale_factor=0.8)

        self.play(self.lecture[2].animate.set_color("#55FF55"))
        self.play(
            Transform(chart, chart3),
            Transform(params, params3),
            run_time=2
        )
        self.play(Indicate(chart))
        self.wait(3)
