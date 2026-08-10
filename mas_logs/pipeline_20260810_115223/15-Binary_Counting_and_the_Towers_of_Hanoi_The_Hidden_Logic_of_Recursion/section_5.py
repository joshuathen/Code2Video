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
        self.setup_layout("Synthesis & Summary", [
            "The puzzle is a physical binary counter.",
            "Three disks require two cubed minus one moves.",
            "Recursive problem solving is binary logic."
        ])
        
        # Elements
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        puzzle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/puzzle.svg")
        
        eq_text = MathTex("3 \\text{ Disks} = 2^3 - 1 = 7 \\text{ steps}", font_size=36)
        label_a = Text("Binary Counting", font_size=28, color=BLUE)
        label_b = Text("Recursive Problem Solving", font_size=28, color=GREEN)
        
        # Layouts
        # 1. Equation + Icon
        eq_group = VGroup(eq_text, disk_icon).arrange(RIGHT, buff=0.3)
        self.place_in_area(eq_group, "B3", "C6", scale_factor=0.9)
        
        # 2. Labels
        self.place_at_grid(label_a, "D3", scale_factor=0.7)
        self.place_at_grid(label_b, "E3", scale_factor=0.7)
        
        # 3. Connection
        connection = DashedLine(label_a.get_bottom(), label_b.get_top(), color=WHITE)
        
        # 4. Hidden Puzzle Icon
        puzzle_icon.scale(0.5).move_to(self.grid["F5"])
        puzzle_icon.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(eq_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(label_a), Create(label_b), Create(connection))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(
            label_a.animate.set_color(YELLOW),
            label_b.animate.set_color(YELLOW),
            puzzle_icon.animate.set_opacity(1),
            run_time=1.5
        )
        self.wait(2)
