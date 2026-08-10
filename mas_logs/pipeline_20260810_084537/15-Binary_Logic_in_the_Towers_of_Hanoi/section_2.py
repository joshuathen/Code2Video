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
        self.setup_layout("Prerequisite: Understanding Binary Counting", 
                          ["Counting in binary: 001, 010, 011.", 
                           "Adding one flips the rightmost bits.", 
                           "The 4-bit flips every four steps."])

        # Initialize Bits
        bit_header = VGroup(*[Text(f"{i}", font_size=36) for i in ["4", "2", "1"]]).arrange(RIGHT, buff=0.8)
        binary_digits = VGroup(*[Text("0", font_size=48, color=YELLOW) for _ in range(3)]).arrange(RIGHT, buff=1.0)
        
        container = VGroup(bit_header, binary_digits).arrange(DOWN, buff=0.5)
        
        self.place_at_grid(container, 'E3', scale_factor=0.6)
        self.place_at_grid(binary_digits, 'E2', scale_factor=0.9)
        self.place_at_grid(bit_header, 'E1', scale_factor=0.8)
        self.add(container)

        def update_bits(val):
            binary_str = format(val, '03b')
            for i in range(3):
                binary_digits[i].set_text(binary_str[i])
                binary_digits[i].set_color(YELLOW if binary_str[i] == '1' else GREY)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        for i in range(1, 4):
            update_bits(i)
            self.wait(0.5)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        update_bits(4)
        self.wait(1)
