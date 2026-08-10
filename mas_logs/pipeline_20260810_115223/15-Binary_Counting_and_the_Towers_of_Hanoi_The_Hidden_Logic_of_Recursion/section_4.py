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
        self.setup_layout("Algorithmic Symmetry", [
            "Exactly one bit changes each time.",
            "Only one disk moves per step.",
            "The largest disk moves at four."
        ])
        
        # Binary representations
        bits = ["001", "010", "011", "100"]
        binary_mobs = VGroup(*[Text(b, font_size=36, color=WHITE) for b in bits])
        binary_mobs.arrange(DOWN, buff=0.4)
        
        # Applying layout fix: Issue 32 & 39
        self.place_in_area(binary_mobs, 'A3', 'D6', scale_factor=0.6)

        # Asset: disk.svg
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        disk_icon.scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(FadeIn(binary_mobs))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Highlight bit change
        change_highlight = SurroundingRectangle(binary_mobs[2][1], color="#00FF00", buff=0.05)
        self.play(Create(change_highlight))
        
        # Map bit to disk (Asset integration)
        arrow = Arrow(start=binary_mobs[2][1].get_right(), end=binary_mobs[2][1].get_right() + RIGHT*0.5, color="#00FF00")
        disk_placed = disk_icon.copy().next_to(arrow, RIGHT)
        self.play(Create(arrow), FadeIn(disk_placed))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        label = Text("Bit 3 = Disk 3", font_size=24, color="#FFFF00")
        
        # Applying layout fix: Issue 31 & 39
        self.place_at_grid(label, 'E4', scale_factor=0.7)
        self.play(Write(label))
        self.wait(2)
