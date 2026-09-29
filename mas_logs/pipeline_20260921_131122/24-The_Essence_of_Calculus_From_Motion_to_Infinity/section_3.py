from manim import *

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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Integration is the inverse of differentiation.",
            "We sum infinitely many thin rectangles.",
            "To find the area under a curve.",
            "It measures total accumulation over time.",
            "Like total food consumed during the day."
        ]
        self.setup_layout("The Integral: The Summation of Slices", lecture_lines)
        
        # Define objects once
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False}, x_length=4, y_length=3)
        curve = axes.plot(lambda x: 0.1 * x**2 + 1, x_range=[0, 4], color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_in_area(VGroup(axes, curve), "A3", "C5", scale_factor=1.0)
        self.play(Create(axes), Create(curve))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Pre-create rectangles for efficiency
        rects = VGroup(*[
            axes.get_riemann_rectangles(curve, x_range=[0, 4], dx=0.5, stroke_width=0.5, color=PURPLE)
        ])
        self.play(Create(rects))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        area = axes.get_area(curve, x_range=[0, 4], color=BLUE, opacity=0.3)
        self.play(ReplacementTransform(rects, area))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        # Use DecimalNumber instead of always_redraw to save resources
        counter = DecimalNumber(0, num_decimal_places=1, color=YELLOW)
        self.place_at_grid(counter, "D3")
        self.play(counter.animate.set_value(5.0), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        integral_sign = MathTex(r"\\int", color="#00FFFF").scale(2)
        # Using placeholder for SVG if real file is missing during test, but adhering to prompt requirement
        try:
            food_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/food.svg")
        except:
            food_icon = Circle(radius=0.3, color=RED).add(Text("F", font_size=20))
            
        self.place_at_grid(integral_sign, "E2")
        self.place_at_grid(food_icon, "E4", scale_factor=0.5)
        self.play(Write(integral_sign), FadeIn(food_icon))
        self.wait(1)
