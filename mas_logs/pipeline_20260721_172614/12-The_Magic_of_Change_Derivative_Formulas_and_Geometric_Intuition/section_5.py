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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        title_str = "Visualizing the Formula: The Growing Square"
        lecture_lines = [
            "Why does the derivative of x squared work?",
            "Consider a square with a side length of x.",
            "If we increase x slightly, the area grows.",
            "The extra area forms two thin rectangles.",
            "These strips represent the two x rate of change."
        ]
        self.setup_layout(title_str, lecture_lines)
        
        # Define Colors
        BLUE_MAIN = "#58C4DD"
        WHITE_MAIN = "#FFFFFF"
        LIGHT_BLUE_DX = "#ACE3EE"
        YELLOW_STRIP = "#FFD700"
        GREEN_FORMULA = "#87FF65"

        # === Animation for Lecture Line 1 ===
        # Draw a blue square (#58C4DD) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg] with sides labeled 'x'.
        self.lecture[0].set_color(BLUE_MAIN)
        
        # main_square is placed at D4 (center (3.5, -0.8))
        # Use SVGMobject asset
        main_square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        main_square.set_color(BLUE_MAIN)
        main_square.set_fill(BLUE_MAIN, opacity=0.3)
        # Scale SVGMobject to a standard side length of 1.8 for grid consistency
        main_square.height = 1.8
        main_square.width = 1.8
        self.place_at_grid(main_square, 'D4')
        
        # Labels adjusted per VideoCritic issues
        x_label_top = MathTex("x", color=WHITE_MAIN)
        self.place_at_grid(x_label_top, 'B4', scale_factor=0.8)
        
        x_label_left = MathTex("x", color=WHITE_MAIN)
        self.place_at_grid(x_label_left, 'D3', scale_factor=0.8)
        
        self.play(
            Create(main_square),
            Write(x_label_top),
            Write(x_label_left),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The text 'Area = x^2' appears inside the square in white (#FFFFFF).
        self.lecture[1].set_color(WHITE_MAIN)
        
        area_text = MathTex("\\text{Area} = x^2", color=WHITE_MAIN)
        self.place_at_grid(area_text, 'D4', scale_factor=0.7)
        
        self.play(Write(area_text), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Expand the top and right edges of the square by a small thickness labeled 'dx' in light blue (#ACE3EE).
        self.lecture[2].set_color(LIGHT_BLUE_DX)
        
        # right_strip placed at D5 (center (4.5, -0.8))
        right_strip = Rectangle(width=0.2, height=1.8, color=LIGHT_BLUE_DX, fill_opacity=0.3)
        self.place_at_grid(right_strip, 'D5')
        
        # top_strip placed at C4 (center (3.5, 0.2))
        top_strip = Rectangle(width=1.8, height=0.2, color=LIGHT_BLUE_DX, fill_opacity=0.3)
        self.place_at_grid(top_strip, 'C4')
        
        # corner_square placed at C5 (center (4.5, 0.2))
        corner_square = Square(side_length=0.2, color=LIGHT_BLUE_DX, fill_opacity=0.3)
        self.place_at_grid(corner_square, 'C5')
        
        # Labels adjusted per VideoCritic issues
        dx_label_top = MathTex("dx", color=LIGHT_BLUE_DX)
        self.place_at_grid(dx_label_top, 'B5', scale_factor=0.6)
        
        dx_label_right = MathTex("dx", color=LIGHT_BLUE_DX)
        self.place_at_grid(dx_label_right, 'D6', scale_factor=0.6)
        
        self.play(
            FadeIn(right_strip),
            FadeIn(top_strip),
            FadeIn(corner_square),
            Write(dx_label_top),
            Write(dx_label_right),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the two new rectangular regions on the edges in bright yellow (#FFD700).
        self.lecture[3].set_color(YELLOW_STRIP)
        
        self.play(
            right_strip.animate.set_color(YELLOW_STRIP).set_fill(opacity=0.6),
            top_strip.animate.set_color(YELLOW_STRIP).set_fill(opacity=0.6),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The formula 'd/dx (x^2) = 2x' fades in at the bottom in light green (#87FF65).
        self.lecture[4].set_color(GREEN_FORMULA)
        
        formula = MathTex("\\frac{d}{dx}(x^2) = 2x", color=GREEN_FORMULA)
        self.place_in_area(formula, 'F1', 'F6', scale_factor=1.0)
        
        self.play(FadeIn(formula, shift=UP), run_time=1.5)
        self.wait(3)
