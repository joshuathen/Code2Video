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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Disk moves follow binary counts.",
            "The first '1' indicates the move.",
            "Binary numbers map to disk moves.",
            "Sequence links binary to disk actions.",
            "Patterns reveal the underlying logic."
        ]
        self.setup_layout("The Binary Connection", lecture_lines)
        
        # Assets
        disk_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # === Animation for Lecture Line 1 ===
        # Represent 3-disk solution as '111' in binary.
        binary_label = Text("Binary: 111", font_size=30, color=BLUE)
        disk = SVGMobject(disk_asset, color=BLUE)
        self.place_at_grid(binary_label, "B2", scale_factor=0.8)
        self.place_at_grid(disk, "B3", scale_factor=0.5)
        self.play(FadeIn(binary_label), FadeIn(disk))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Animate '111' becoming '7' in decimal.
        decimal_label = Text("Decimal: 7", font_size=30, color=YELLOW)
        self.place_at_grid(decimal_label, "B6", scale_factor=0.8)
        self.play(Transform(binary_label.copy(), decimal_label))
        self.play(FadeIn(decimal_label))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Draw green (#32CD32) path connecting binary to disks
        path = Line(binary_label.get_right(), disk.get_left(), color="#32CD32")
        self.play(Create(path))
        self.lecture[2].set_color("#32CD32")

        # === Animation for Lecture Line 4 ===
        # Highlight disk movement sync with binary sequence.
        disk_move = Text("Move Disk", font_size=30, color=RED)
        self.place_in_area(disk_move, "E2", "E4", scale_factor=0.75)
        self.play(FadeIn(disk_move))
        self.lecture[3].set_color(RED)

        # === Animation for Lecture Line 5 ===
        # Patterns reveal the underlying logic.
        logic_label = Text("Logic: 2^n - 1", font_size=24, color=PURPLE)
        self.place_at_grid(logic_label, "D5", scale_factor=0.7)
        self.play(Write(logic_label))
        self.lecture[4].set_color(PURPLE)

        self.wait(2)
