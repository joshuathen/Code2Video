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
        
        # Objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2 + 1, x_range=[0, 4])
        area = axes.get_area(curve, x_range=[0, 4], color=GREEN, opacity=0.5)
        rects = VGroup(*[
            Rectangle(height=axes.c2p(0, 0.5 * x**2 + 1)[1] - axes.c2p(0,0)[1], 
                      width=0.4, color=WHITE, fill_opacity=0.3)
            for x in np.arange(0, 4, 0.4)
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(VGroup(axes, curve), "A2", "C5", scale_factor=0.5)
        self.play(Create(axes), Create(curve))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        # Initialize rectangles at the bottom of the curve
        for i, rect in enumerate(rects):
            x = i * 0.4
            rect.move_to(axes.c2p(x + 0.2, (0.5 * x**2 + 1)/2))
        self.play(Create(rects))
        # Shrinking animation logic handled by count increase (simplified here)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.play(ReplacementTransform(rects, area))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        val = ValueTracker(0)
        label = always_redraw(lambda: MathTex(f"{val.get_value():.1f}").move_to(self.grid["E3"]))
        self.add(label)
        self.play(val.animate.set_value(5.0), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        integral_sign = MathTex(r"\int").scale(2)
        food_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/food.svg")
        self.place_at_grid(integral_sign, "E2")
        self.place_at_grid(food_icon, "E4", scale_factor=0.5)
        self.play(Write(integral_sign), FadeIn(food_icon))
        self.wait(2)
