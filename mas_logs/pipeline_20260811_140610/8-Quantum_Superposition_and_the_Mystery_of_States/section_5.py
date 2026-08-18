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
        self.setup_layout("Application: Quantum Computing", [
            "Superposition allows for parallel computation.",
            "Classical bits test paths one by one.",
            "Quantum computers process all paths simultaneously."
        ])
        
        # Assets
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer, 'C1', scale_factor=0.5)

        # Paths
        paths = VGroup()
        for i in range(3):
            path = Line(start=computer.get_right() + RIGHT*0.2, end=computer.get_right() + RIGHT*5 + UP*(1-i), color=WHITE)
            paths.add(path)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(computer), Create(paths))
        self.play(self.lecture[0].animate.set_color("#87CEEB"))

        # === Animation for Lecture Line 2 ===
        bits = [Dot(color="#FF0000").move_to(p.get_start()) for p in paths]
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        
        for bit in bits:
            self.play(MoveAlongPath(bit, paths[bits.index(bit)]), run_time=1)
            self.play(FadeOut(bit))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(paths.animate.set_color("#00FF00"))
        
        computer_end = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer_end, 'C6', scale_factor=0.5)
        self.play(FadeIn(computer_end))
        self.wait(2)
