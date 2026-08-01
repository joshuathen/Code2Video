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
        # Initial Setup
        title = "The Beauty of Unity"
        lines = [
            'Adding one to both sides equals zero.', 
            'Five fundamental constants unite in one elegant equation.', 
            'This is the most beautiful identity in mathematics.'
        ]
        self.setup_layout(title, lines)

        # Define colors for constants
        color_e = YELLOW
        color_pi = BLUE
        color_i = PURPLE
        color_one = GREEN
        color_zero = RED

        # === Animation for Lecture Line 1 ===
        # Create components for e^(pi i) = -1
        # Using Text to avoid LaTeX environment issues as per team standard
        e1 = Text("e", color=color_e)
        pi1 = Text("π", color=color_pi).scale(0.7)
        i1 = Text("i", color=color_i).scale(0.7)
        eq1_sym = Text("=", color=WHITE)
        minus1 = Text("-", color=WHITE)
        one1 = Text("1", color=color_one)

        # Arrange superscript
        exponent1 = VGroup(pi1, i1).arrange(RIGHT, buff=0.05)
        exponent1.next_to(e1.get_top(), RIGHT, buff=-0.1).shift(UP*0.1)
        base_eq1 = VGroup(e1, exponent1)
        
        eq_start = VGroup(base_eq1, eq1_sym, minus1, one1).arrange(RIGHT, buff=0.3)
        self.place_in_area(eq_start, "B1", "D6", scale_factor=1.5)

        # Create components for e^(pi i) + 1 = 0
        e2 = Text("e", color=color_e)
        pi2 = Text("π", color=color_pi).scale(0.7)
        i2 = Text("i", color=color_i).scale(0.7)
        plus2 = Text("+", color=WHITE)
        one2 = Text("1", color=color_one)
        eq2_sym = Text("=", color=WHITE)
        zero2 = Text("0", color=color_zero)

        exponent2 = VGroup(pi2, i2).arrange(RIGHT, buff=0.05)
        exponent2.next_to(e2.get_top(), RIGHT, buff=-0.1).shift(UP*0.1)
        base_eq2 = VGroup(e2, exponent2)
        
        eq_end = VGroup(base_eq2, plus2, one2, eq2_sym, zero2).arrange(RIGHT, buff=0.3)
        self.place_in_area(eq_end, "B1", "D6", scale_factor=1.5)

        # Highlight first lecture line and perform transform
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(eq_start))
        self.wait(1)
        self.play(ReplacementTransform(eq_start, eq_end))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Sequentially scale and flash 'e', 'π', 'i', '1', and '0'
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        constants_to_flash = [e2, pi2, i2, one2, zero2]
        for const in constants_to_flash:
            self.play(
                const.animate.scale(1.3),
                Flash(const, color=const.get_color(), flash_radius=0.5),
                run_time=0.6
            )
            self.play(const.animate.scale(1/1.3), run_time=0.4)

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw a glowing golden (#FFD700) border around the entire identity
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        golden_rect = SurroundingRectangle(eq_end, color="#FFD700", buff=0.4, stroke_width=4)
        # Simulate glow with a second rectangle
        glow = golden_rect.copy().set_stroke(width=8, opacity=0.3)
        
        self.play(Create(golden_rect))
        self.play(FadeIn(glow))
        self.play(glow.animate.scale(1.1).set_opacity(0), rate_func=there_and_back, run_time=2)
        
        self.wait(3)
