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
        lecture_lines = ["Moves follow 2^n - 1.", "3 disks need 7 moves total.", "Binary limits match Hanoi moves."]
        self.setup_layout("The Power of 2^n - 1", lecture_lines)
        
        # Assets
        disk = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        
        # Objects
        formula = MathTex("2^n - 1").set_color(BLUE)
        disk_label = Text("n = 3").set_color(YELLOW)
        moves_label = Text("Moves = 7").set_color(GREEN)
        labels_group = VGroup(disk_label, moves_label).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(formula, 'B3', 'B5', scale_factor=1.2)
        disk_icon = self.place_at_grid(disk.copy(), 'B2', scale_factor=0.5)
        self.play(Write(formula), FadeIn(disk_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_in_area(labels_group, 'D3', 'D4', scale_factor=1.0)
        disk_icon2 = self.place_at_grid(disk.copy(), 'D2', scale_factor=0.5)
        self.play(FadeIn(labels_group), FadeIn(disk_icon2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        binary_label = Text("Binary 111 = 7").set_color(YELLOW)
        self.place_at_grid(binary_label, 'E3', scale_factor=0.9)
        disk_icon3 = self.place_at_grid(disk.copy(), 'E5', scale_factor=0.5)
        self.play(Write(binary_label), FadeIn(disk_icon3))
        self.wait(2)
