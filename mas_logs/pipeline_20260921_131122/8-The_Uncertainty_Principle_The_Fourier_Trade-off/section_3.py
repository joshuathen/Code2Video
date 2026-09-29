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
        lecture_lines = ["Time and frequency widths are reciprocal.", "Widths are linked by a constant.", "This is an inherent wave property.", "Shrink time, frequency side grows.", "The balance is mathematically enforced."]
        self.setup_layout("The Core Concept: The Reciprocal Relationship", lecture_lines)
        
        # Setup visual elements
        time_bar = Rectangle(height=0.5, width=4, color=WHITE, fill_opacity=0.6)
        freq_bar = Rectangle(height=0.5, width=4, color=WHITE, fill_opacity=0.6)
        
        # Applying requested position/scale fixes (Issues 26, 27)
        self.place_at_grid(time_bar, "B3", scale_factor=0.6)
        self.place_at_grid(freq_bar, "E3", scale_factor=0.6)
        
        time_label = Text("Δt", font_size=24).next_to(time_bar, LEFT)
        freq_label = Text("Δf", font_size=24).next_to(freq_bar, LEFT)
        
        # Asset loading
        balance_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balance.svg")
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        self.place_at_grid(balance_icon, "A3", scale_factor=0.5)
        
        formula = MathTex(r"\Delta t \cdot \Delta f \geq C", font_size=32, color=GREEN)
        self.place_in_area(formula, "C4", "D5", scale_factor=0.9) # Fix issue 28

        # === Animation for Lecture Line 1 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            Write(time_bar), 
            Write(freq_bar), 
            Write(time_label), 
            Write(freq_label),
            FadeIn(balance_icon)
        )

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN), Write(formula))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))

        # === Animation for Lecture Line 4 ===
        # Shrink time_bar, grow freq_bar
        self.play(
            self.lecture[3].animate.set_color("#FF00FF"),
            time_bar.animate.set_width(1.5),
            freq_bar.animate.set_width(6.0),
            run_time=2
        )

        # === Animation for Lecture Line 5 ===
        # Fading out items (Issue 18)
        self.play(
            self.lecture[4].animate.set_color(YELLOW),
            FadeOut(time_bar), 
            FadeOut(freq_bar), 
            FadeOut(time_label), 
            FadeOut(freq_label), 
            FadeOut(formula),
            FadeOut(balance_icon),
            FadeIn(scale_icon.move_to(self.grid["C3"]).scale(0.8))
        )
        self.wait(2)
