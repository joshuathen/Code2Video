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
            "Binary steps map to disk moves.",
            "Disk one moves every odd step.",
            "Disk two moves every two steps.",
            "Disk three moves every four steps.",
            "Binary counting guides the movement pattern."
        ]
        self.setup_layout("Mapping Binary to Moves", lecture_lines)
        
        # Assets
        disk_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # Colors
        colors = [BLUE, GREEN, YELLOW, RED, PURPLE]
        for i, line in enumerate(self.lecture):
            line.set_color(colors[i])

        # Animation Setup
        bin_str = Text("001", font_size=40, color=WHITE)
        self.place_at_grid(bin_str, 'B3', scale_factor=1.2)
        
        disk1 = SVGMobject(disk_asset, color=BLUE)
        disk2 = SVGMobject(disk_asset, color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(bin_str))
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(disk1, 'C2', scale_factor=0.8)
        self.play(FadeIn(disk1), disk1.animate.set_color("#FF00FF"))
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        bin_str2 = Text("010", font_size=40, color=WHITE)
        self.place_at_grid(bin_str2, 'B4', scale_factor=1.0)
        self.play(Transform(bin_str, bin_str2))
        self.lecture[2].set_color(colors[2])
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(disk2, 'C4', scale_factor=0.8)
        self.play(FadeIn(disk2), disk2.animate.set_color("#FF00FF"))
        self.lecture[3].set_color(colors[3])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        binary_label = Text("Binary Counter", font_size=20, color=WHITE)
        self.place_at_grid(binary_label, 'A2', scale_factor=0.9)
        self.play(Write(binary_label))
        self.lecture[4].set_color(colors[4])
        self.wait(2)
