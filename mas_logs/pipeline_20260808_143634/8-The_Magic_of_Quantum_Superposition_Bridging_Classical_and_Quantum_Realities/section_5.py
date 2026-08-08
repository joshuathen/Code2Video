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
        lecture_lines = [
            "Superposition powers quantum computing's parallel advantage.",
            "Classical bits process sequentially, quantum bits parallel.",
            "Quantum computers solve complex problems instantly."
        ]
        self.setup_layout("Quantum Advantage: Why Superposition Matters", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Classical bit (mouse) path animation using icon
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        mouse = Dot(color=BLUE)
        path = Line(start=self.grid['B4'], end=self.grid['B6'], color=GRAY)
        self.add(path)
        self.place_at_grid(mouse, 'B4', scale_factor=0.6)
        self.place_at_grid(computer, 'A4', scale_factor=0.3)
        self.play(mouse.animate.move_to(self.grid['B6']), run_time=2)
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Quantum bit (superposition of paths) using icon
        microchip_q = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg")
        q_mouse = Dot(color=YELLOW)
        branches = VGroup(*[Line(start=self.grid['C4'], end=self.grid['C6'], color=YELLOW) for _ in range(3)])
        branches.arrange(DOWN, buff=0.1)
        self.add(branches)
        self.place_at_grid(q_mouse, 'C4', scale_factor=0.6)
        self.place_at_grid(microchip_q, 'C2', scale_factor=0.3)
        self.play(q_mouse.animate.move_to(self.grid['C6']), run_time=2)
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Computational power highlight
        microchip_perf = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg")
        box = Square(side_length=1.5, color="#00FF00")
        self.place_in_area(box, 'A3', 'C5', scale_factor=0.7)
        self.place_at_grid(microchip_perf, 'B4', scale_factor=0.5)
        self.play(Create(box), FadeIn(microchip_perf))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
