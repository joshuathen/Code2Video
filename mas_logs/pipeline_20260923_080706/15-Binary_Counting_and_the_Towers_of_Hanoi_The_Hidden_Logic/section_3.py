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
            "Disk moves mirror binary counting patterns.",
            "Binary steps determine which disk moves.",
            "Example: Step 1 moves disk 1.",
            "Step 2 moves disk 2.",
            "Step 3 moves disk 1 again."
        ]
        self.setup_layout("The Binary Connection", lecture_lines)
        
        # Assets
        disk_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        rod_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg"

        # Setup visual elements
        binary_label = Text("001", color="#00FF00", font_size=40)
        # Fix 26: Place binary_label at 'D3'
        self.place_at_grid(binary_label, "D3", scale_factor=1.0)
        
        disk = SVGMobject(disk_svg, color=BLUE, fill_opacity=0.5)
        # Fix 27: Place disk at 'E4'
        self.place_at_grid(disk, "E4", scale_factor=1.2)

        # Fix 28: Group and re-position
        group = VGroup(binary_label, disk)
        self.place_in_area(group, "C2", "E2", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(group))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        binary_label2 = Text("010", color="#00FFFF", font_size=40).move_to(binary_label.get_center())
        self.play(ReplacementTransform(binary_label, binary_label2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        disk_move = disk.copy()
        self.play(disk_move.animate.shift(UP * 1.5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        binary_label3 = Text("011", color="#FFFF00", font_size=40).move_to(binary_label2.get_center())
        self.play(ReplacementTransform(binary_label2, binary_label3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        self.play(FadeOut(binary_label3), FadeOut(disk), FadeOut(disk_move))
        self.wait(1)
