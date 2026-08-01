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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initialize the layout
        self.setup_layout("The Product Rule: Growing Areas", [
            "The derivative of a product isn't just multiplying derivatives.", 
            "Visualize a rectangle with growing width and height.", 
            "The change in area depends on both side's rates.", 
            "We add the growth from each side separately.", 
            "The rule: derivative of first times second plus vice versa."
        ])
        
        # Define hex colors
        u_color = "#1E90FF"    # Blue
        v_color = "#FF69B4"    # Pink
        udv_color = "#FFFF00"  # Yellow
        vdu_color = "#00FF00"  # Green
        white = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        wrong_rule = MathTex(
            "\\frac{d}{dx}(u \\cdot v) \\neq \\frac{du}{dx} \\cdot \\frac{dv}{dx}", 
            color=WHITE
        )
        self.place_in_area(wrong_rule, 'B1', 'D6', scale_factor=1.2)
        self.play(Write(wrong_rule))
        self.wait(1.5)
        self.play(FadeOut(wrong_rule))

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Base rectangle
        main_rect = Rectangle(width=2, height=2, stroke_color=WHITE, stroke_width=2, fill_opacity=0.2, fill_color=WHITE)
        rect_group = VGroup(main_rect)
        self.place_in_area(rect_group, 'B2', 'D4', scale_factor=1.0)
        
        # Dimension labels - using grid positions requested in Issue 26
        # To align with requested grid: u_label (C1) is height side, v_label (A3) is width side
        u_label = MathTex("u", color=u_color)
        v_label = MathTex("v", color=v_color)
        self.place_at_grid(u_label, 'C1', scale_factor=0.8) # Left side
        self.place_at_grid(v_label, 'A3', scale_factor=0.8) # Top side
        
        self.play(Create(main_rect), Write(u_label), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Growth increments
        # du rect (width growth) - attaches to side
        du_rect = Rectangle(width=0.4, height=2, fill_opacity=0.4, fill_color=vdu_color, stroke_width=1, stroke_color=vdu_color)
        du_rect.next_to(main_rect, RIGHT, buff=0)
        
        # dv rect (height growth) - attaches to top
        dv_rect = Rectangle(width=2, height=0.4, fill_opacity=0.4, fill_color=udv_color, stroke_width=1, stroke_color=udv_color)
        dv_rect.next_to(main_rect, UP, buff=0)
        
        # Corner rect
        corner_rect = Rectangle(width=0.4, height=0.4, fill_opacity=0.2, fill_color=WHITE, stroke_width=0)
        corner_rect.next_to(du_rect, UP, buff=0)
        
        du_text = MathTex("du", color=vdu_color)
        dv_text = MathTex("dv", color=udv_color)
        self.place_at_grid(du_text, 'C5', scale_factor=0.7)
        self.place_at_grid(dv_text, 'A5', scale_factor=0.7)
        
        self.play(
            FadeIn(du_rect), FadeIn(dv_rect), FadeIn(corner_rect),
            Write(du_text), Write(dv_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Emphasize growth areas
        v_du_label = MathTex("v \\cdot du", color=vdu_color)
        u_dv_label = MathTex("u \\cdot dv", color=udv_color)
        
        # Center them in the increment rectangles
        v_du_label.move_to(du_rect.get_center()).scale(0.7)
        u_dv_label.move_to(dv_rect.get_center()).scale(0.7)
        
        self.play(
            du_rect.animate.set_fill(opacity=0.7),
            dv_rect.animate.set_fill(opacity=0.7),
            Write(v_du_label),
            Write(u_dv_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Final Formula and Example
        formula_latex = VGroup(
            MathTex("(u \\cdot v)' = u \\cdot v' + v \\cdot u'", color=white),
            MathTex("\\text{Example: } (x \\sin x)' = x \\cos x + \\sin x", color=white, font_size=32)
        ).arrange(DOWN, buff=0.4)
        
        # Anchor to grid as per Issue 27
        self.place_in_area(formula_latex, 'E1', 'F6', scale_factor=0.9)
        
        self.play(Write(formula_latex))
        self.wait(3)
