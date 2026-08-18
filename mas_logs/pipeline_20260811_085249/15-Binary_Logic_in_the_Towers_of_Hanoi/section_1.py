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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Nature of Binary", 
                          ["Binary uses only zero and one.", 
                           "Each bit represents a power of two.", 
                           "Counting in binary follows simple patterns."])
        
        # Assets
        icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        
        # Grid Title
        grid_title = Text("Binary Grid Visual", font_size=24)
        self.place_in_area(grid_title, 'A2', 'A5', scale_factor=1.0)
        
        # Grid Dots
        grid_dots = VGroup(*[Dot(radius=0.1) for _ in range(36)])
        grid_dots.arrange_in_grid(rows=6, cols=6, buff=0.3)
        self.place_in_area(grid_dots, 'B2', 'F5', scale_factor=0.9)
        
        # Binary Counter
        bits = VGroup(*[Text(digit, font_size=72) for digit in ["0", "0", "0"]])
        bits.arrange(RIGHT, buff=0.2)
        self.place_at_grid(bits, 'C5', scale_factor=1.2)
        
        # Icon
        icon = SVGMobject(icon_path).scale(0.5)
        self.place_at_grid(icon, 'F1')
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(bits), FadeIn(grid_title), FadeIn(grid_dots), FadeIn(icon))
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 000 -> 001
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        bits1 = VGroup(*[Text(digit, font_size=72) for digit in ["0", "0", "1"]])
        bits1.arrange(RIGHT, buff=0.2)
        self.place_at_grid(bits1, 'C5', scale_factor=1.2)
        bits1[2].set_color("#FF4500")
        
        self.play(Transform(bits, bits1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # 010
        bits2 = VGroup(*[Text(digit, font_size=72) for digit in ["0", "1", "0"]])
        bits2.arrange(RIGHT, buff=0.2)
        self.place_at_grid(bits2, 'C5', scale_factor=1.2)
        bits2[1].set_color("#FF4500")
        
        self.play(Transform(bits, bits2))
        self.wait(1)
