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
        lecture_lines = [
            "Binary uses only 0s and 1s.",
            "Each position represents a power of two.",
            "Combine powers to form any number.",
            "Example: 101 means 4 plus 1.",
            "Binary simplifies complex counting systems."
        ]
        self.setup_layout("Prerequisite: The Language of Binary", lecture_lines)
        
        # Assets
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        counter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/counter.svg")
        
        self.place_at_grid(computer_icon, "A3", scale_factor=0.5)
        self.add(computer_icon)

        # Binary display: 3 bits
        bits = VGroup(*[Text("0", color=GREY, font_size=48) for _ in range(3)])
        # FIX: Adjusted position to B3
        self.place_at_grid(bits, "B3", scale_factor=1.0)
        bits.arrange(RIGHT, buff=0.5)
        self.add(bits)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        # Indicate powers
        powers = VGroup(*[Text(p, color=BLUE, font_size=20) for p in ["4", "2", "1"]])
        # FIX: Adjusted position to C3 and scale 0.9
        self.place_at_grid(powers, "C3", scale_factor=0.9)
        powers.arrange(RIGHT, buff=0.6)
        self.play(Create(powers))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF5733"))
        # Update text content safely
        bits[0].become(Text("1", color=YELLOW, font_size=48).move_to(bits[0].get_center()))
        bits[1].become(Text("0", color=GREY, font_size=48).move_to(bits[1].get_center()))
        bits[2].become(Text("1", color=YELLOW, font_size=48).move_to(bits[2].get_center()))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF5733"))
        self.place_at_grid(counter_icon, "A5", scale_factor=0.5)
        self.add(counter_icon)
        
        decimal_val = Text("Value: 5", color=WHITE, font_size=32)
        # FIX: Adjusted position to D3
        self.place_at_grid(decimal_val, "D3", scale_factor=1.0)
        self.play(Write(decimal_val))
        self.wait(2)
