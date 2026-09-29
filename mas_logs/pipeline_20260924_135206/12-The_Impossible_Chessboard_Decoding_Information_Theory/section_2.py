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
        lecture_lines = [
            "64 squares are represented by 6 binary bits.",
            "Bit positions 0 through 63 define every square.",
            "XOR parity acts as our digital checksum."
        ]
        self.setup_layout("Prerequisite: Binary Encoding", lecture_lines)
        
        # Load asset
        chessboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chessboard.svg")
        self.place_in_area(chessboard, "A2", "C5", scale_factor=0.5)
        self.add(chessboard)
        
        # === Animation for Lecture Line 1 ===
        # Display binary numbers 0 and 1. Color: #00FFFF (0), #FF00FF (1).
        bit0 = Text("0", font_size=40, color="#00FFFF")
        bit1 = Text("1", font_size=40, color="#FF00FF")
        self.place_at_grid(bit0, "B2", scale_factor=0.8)
        self.place_at_grid(bit1, "B5", scale_factor=0.8)
        self.play(Write(bit0), Write(bit1))
        self.lecture[0].set_color("#FFFF00") 
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualize bit sequence construction.
        sequence = VGroup(*[Text(str(x), color="#FFFFFF") for x in [1, 0, 1, 0, 0, 1]]).arrange(RIGHT, buff=0.2)
        self.place_in_area(sequence, "C2", "C5", scale_factor=1.0)
        self.play(FadeIn(sequence))
        self.lecture[1].set_color("#FFFF00") 
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the parity bit concept.
        parity_bit = Text("1", color="#FF5733")
        parity_label = Text("Parity", font_size=18, color="#FF5733")
        parity_group = VGroup(parity_bit, parity_label).arrange(DOWN)
        self.place_at_grid(parity_group, "D4", scale_factor=0.9)
        
        self.play(Write(parity_group), sequence.animate.set_color("#FF5733"))
        self.lecture[2].set_color("#FFFF00") 
        self.wait(2)
