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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout("Scaling to y and Summary", [
            "Apply the same logic to solve for y.",
            "Swap the second column with target vector b.",
            "Cramer's Rule turns algebra into geometric ratios."
        ])
        
        # Colors for consistency
        v1_color = "#ADD8E6"  # Light Blue
        v2_color = "#90EE90"  # Light Green
        b_color = "#FFD700"   # Gold
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(v1_color)
        
        # Setup Axes within the grid area (B2 to F6 as per Issue 34)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": GREY, "include_tip": True}
        )
        self.place_in_area(axes, "B2", "F6")
        
        # Define vectors based on x=1, y=2 solution (v1=[2,0], v2=[1,2] -> b=[4,4])
        v1 = Vector(axes.c2p(2, 0) - axes.c2p(0, 0), color=v1_color)
        v1.shift(axes.c2p(0, 0))
        
        v2 = Vector(axes.c2p(1, 2) - axes.c2p(0, 0), color=v2_color)
        v2.shift(axes.c2p(0, 0))
        
        v1_label = MathTex(r"\vec{v}_1", color=v1_color).scale(0.8)
        v1_label.next_to(v1.get_end(), DOWN, buff=0.1)
        
        v2_label = MathTex(r"\vec{v}_2", color=v2_color).scale(0.8)
        v2_label.next_to(v2.get_end(), LEFT, buff=0.1)

        # Original Area Parallelogram (v1, v2)
        poly_orig = Polygon(
            axes.c2p(0,0), axes.c2p(2,0), axes.c2p(3,2), axes.c2p(1,2),
            fill_opacity=0.3, fill_color=v1_color, stroke_width=1, stroke_color=v1_color
        )

        self.play(Write(axes))
        self.play(GrowArrow(v1), FadeIn(v1_label), GrowArrow(v2), FadeIn(v2_label))
        self.play(FadeIn(poly_orig))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(b_color)
        
        # Target vector b (replaces v2 in the visual logic for y)
        b = Vector(axes.c2p(4, 4) - axes.c2p(0, 0), color=b_color)
        b.shift(axes.c2p(0, 0))
        b_label = MathTex(r"\vec{b}", color=b_color).scale(0.8)
        b_label.next_to(b.get_end(), UP, buff=0.1)
        
        # New Area Parallelogram (v1, b) for determining y
        poly_y = Polygon(
            axes.c2p(0,0), axes.c2p(2,0), axes.c2p(4,4), axes.c2p(2,4),
            fill_opacity=0.3, fill_color=b_color, stroke_width=1, stroke_color=b_color
        )

        self.play(
            FadeOut(v2), 
            FadeOut(v2_label),
            GrowArrow(b),
            FadeIn(b_label),
            ReplacementTransform(poly_orig, poly_y)
        )
        
        # Display the resulting y value ratio (A4 to A6 as per Issue 33)
        y_val_text = MathTex(r"y = \frac{\text{Area}(\vec{v}_1, \vec{b})}{\text{Area}(\vec{v}_1, \vec{v}_2)} = 2", color=WHITE)
        self.place_in_area(y_val_text, "A4", "A6", scale_factor=0.7)
        self.play(Write(y_val_text))
        
        # Animate Vector-Bot reaching the solution coordinates (1, 2)
        vector_bot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").scale(0.4).set_color(b_color)
        vector_bot.move_to(axes.c2p(0,0))
        
        self.play(FadeIn(vector_bot))
        self.play(vector_bot.animate.move_to(axes.c2p(1,2)), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        self.play(
            FadeOut(axes), FadeOut(v1), FadeOut(b), FadeOut(v1_label), FadeOut(b_label), 
            FadeOut(poly_y), FadeOut(y_val_text), FadeOut(vector_bot)
        )
        
        # Display the general Cramer's Rule formula (B2 to E6, scale 1.2 as per Issue 35)
        final_formula = MathTex(r"x_i = \frac{\det(A_i)}{\det(A)}", color=WHITE)
        self.place_in_area(final_formula, "B2", "E6", scale_factor=1.2)
        
        self.play(Write(final_formula))
        self.wait(3)
