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
            "Towers of Hanoi: A puzzle of three pegs.",
            "Can we predict the movement of disks?",
            "Binary is the language of on and off.",
            "Three disks require seven moves total.",
            "Binary bits mirror the growth of moves."
        ]
        self.setup_layout("Introduction: The Legend and the Binary Link", lecture_lines)
        
        # Define objects
        pegs = VGroup(*[Line(DOWN*0.5, UP*0.5, color=BLUE) for _ in range(3)]).arrange(RIGHT, buff=1.5)
        disks = VGroup(*[Rectangle(height=0.2, width=w, color=color, fill_opacity=1) 
                        for w, color in [(0.8, RED), (0.6, GREEN), (0.4, YELLOW)]]).arrange(DOWN, buff=0)
        towers = VGroup(pegs, disks).arrange(DOWN, buff=0)
        
        binary_on = Text("1 (ON)", color=YELLOW)
        binary_off = Text("0 (OFF)", color=BLUE)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.place_at_grid(towers, 'C3', scale_factor=0.6)))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Simple highlight
        self.play(Indicate(towers))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        self.play(Write(self.place_at_grid(binary_on, 'B5', scale_factor=0.5)))
        self.play(Write(self.place_at_grid(binary_off, 'C5', scale_factor=0.5)))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        count_text = Text("Moves: 7", color=RED)
        self.play(FadeIn(self.place_at_grid(count_text, 'E5', scale_factor=0.8)))
        self.lecture[3].set_color(RED)

        # === Animation for Lecture Line 5 ===
        bits = VGroup(*[Text(str(i), color=WHITE) for i in [0, 1, 1]]).arrange(RIGHT)
        self.play(FadeIn(self.place_at_grid(bits, 'F5', scale_factor=0.7)))
        self.lecture[4].set_color(ORANGE)
        self.wait(2)
