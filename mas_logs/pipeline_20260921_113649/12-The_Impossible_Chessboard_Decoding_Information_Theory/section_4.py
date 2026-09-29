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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Connecting to Information Theory", [
            "This isn't just a simple guess.",
            "We encode information into the board.",
            "Parity acts as an error-correcting code."
        ])
        
        # Visual assets
        board_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/board.svg")
        
        # Message buffer setup (using place_in_area as requested)
        message_buffer = VGroup(board_icon, Text("Message Buffer", font_size=20))
        message_buffer.arrange(DOWN)
        self.place_in_area(message_buffer, 'A2', 'B5', scale_factor=0.9)
        
        # Bits representation
        bits = VGroup(*[Square(side_length=0.2, color=WHITE) for _ in range(4)])
        bits.arrange(RIGHT, buff=0.1)
        bits.move_to(board_icon.get_center())
        
        # Noise channel
        noise_label = Text("Noise Channel", font_size=20, color=GRAY)
        self.place_at_grid(noise_label, 'C3', scale_factor=0.8)
        
        # Parity
        parity_bit = Square(side_length=0.2, color=GREEN)
        parity_label = Text("Parity", font_size=18, color="#00FF00")
        parity_group = VGroup(parity_bit, parity_label).arrange(RIGHT)
        self.place_at_grid(parity_group, 'B6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(message_buffer))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(bits))
        self.play(Write(noise_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(parity_group))
        self.wait(1)
