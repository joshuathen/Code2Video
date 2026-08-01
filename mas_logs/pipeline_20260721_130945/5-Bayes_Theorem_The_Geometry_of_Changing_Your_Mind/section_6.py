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
        # Setup the basic layout with title and lecture lines
        self.setup_layout(
            "From Geometry to Formula", 
            [
                "This geometric process is exactly what Bayes' Theorem describes.",
                "The numerator represents the specific shaded Glitch-Spark area.",
                "Dividing by the total shaded area gives the posterior."
            ]
        )
        
        # Colors
        GLITCH_SPARK_COLOR = "#F1C40F"  # Gold
        HIGHLIGHT_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # Display the equation P(H|E) = [P(E|H)*P(H)]/P(E) and the glitch icon asset.
        
        # Manually construct formula to avoid LaTeX compilation errors seen in previous attempts
        # (L022: Fallback or simplification for MathTex errors)
        lhs = MathTex(r"P(H|E) = ", color=WHITE)
        num = MathTex(r"P(E|H)P(H)", color=WHITE)
        den = MathTex(r"P(E)", color=WHITE)
        fraction_line = Line(LEFT, RIGHT, color=WHITE).set_width(num.width + 0.2)
        rhs = VGroup(num, fraction_line, den).arrange(DOWN, buff=0.1)
        formula = VGroup(lhs, rhs).arrange(RIGHT, buff=0.1)
        
        # [Critic Fix Issue 39]: Position formula at D3 to F4 to avoid lecture overlap
        self.place_in_area(formula, 'D3', 'F4', scale_factor=1.0)
        
        # [Asset Integration Issue 24]: Incorporate glitch icon from provided path
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/glitch.svg]
        glitch_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glitch.svg")
        self.place_at_grid(glitch_icon, 'B4', scale_factor=0.6)
        
        # Highlight lecture line 1
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(
            Write(lhs),
            Write(rhs),
            FadeIn(glitch_icon),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Box the numerator and point to the gold area (#F1C40F).
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GLITCH_SPARK_COLOR)
        
        # Visual representation of the 'True Positive' area (Glitch-Spark)
        tp_rect = Rectangle(
            width=1.2, height=0.7, 
            fill_color=GLITCH_SPARK_COLOR, fill_opacity=0.8, 
            stroke_color=WHITE
        )
        tp_label = Text("Glitch-Spark Area", font_size=16, color=WHITE)
        tp_group = VGroup(tp_rect, tp_label).arrange(DOWN, buff=0.1)
        
        # [Critic Fix Issue 40]: Position tp_group at E2 to separate it from lecture text
        self.place_at_grid(tp_group, 'E2', scale_factor=0.8)
        
        # Create a box around the numerator
        num_box = SurroundingRectangle(num, color=GLITCH_SPARK_COLOR, buff=0.1)
        
        # Arrow connecting the formula numerator to the visual area
        num_arrow = Arrow(
            start=num_box.get_left(),
            end=tp_group.get_right(),
            color=GLITCH_SPARK_COLOR,
            buff=0.1
        )
        
        self.play(
            FadeIn(tp_group),
            Create(num_box),
            GrowArrow(num_arrow)
        )
        # Use Indicate to draw attention (L004)
        self.play(Indicate(num, color=GLITCH_SPARK_COLOR))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Box the denominator and point to the entire width (#FFFFFF).
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Visual representation of the 'Total Shaded' area (TP + FP)
        te_tp = Rectangle(width=0.8, height=0.7, fill_color=GLITCH_SPARK_COLOR, fill_opacity=0.5)
        te_fp = Rectangle(width=0.4, height=0.7, fill_color=BLUE, fill_opacity=0.5)
        te_rects = VGroup(te_tp, te_fp).arrange(RIGHT, buff=0)
        te_rects.set_stroke(WHITE, width=2)
        
        te_label = Text("Total Evidence Area", font_size=16, color=WHITE)
        te_group = VGroup(te_rects, te_label).arrange(DOWN, buff=0.1)
        
        # [Critic Fix Issue 41]: Position te_group at E5 to prevent overlap with denominator
        self.place_at_grid(te_group, 'E5', scale_factor=0.8)
        
        # Create a box around the denominator
        den_box = SurroundingRectangle(den, color=WHITE, buff=0.1)
        
        # Arrow connecting the formula denominator to the total evidence area
        den_arrow = Arrow(
            start=den_box.get_right(),
            end=te_group.get_left(),
            color=WHITE,
            buff=0.1
        )
        
        self.play(
            FadeIn(te_group),
            Create(den_box),
            GrowArrow(den_arrow)
        )
        self.play(Indicate(den, color=WHITE))
        self.wait(3)

        # Final state cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(2)
