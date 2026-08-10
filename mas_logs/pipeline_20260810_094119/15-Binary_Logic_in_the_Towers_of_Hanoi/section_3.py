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
        self.setup_layout("Mapping Moves to Binary Codes", [
            "We label moves from one to two power n.",
            "Binary digits map directly to disc moves.",
            "The lowest set bit determines which disc moves.",
            "[Asset: binary_mapping_table]"
        ])
        
        # Color definitions for animations
        c1 = "#00FFFF" # Cyan
        c2 = "#FF00FF" # Magenta
        c3 = "#808080" # Grey
        
        # Table placeholders
        table = VGroup(
            Text("Move | Binary | Disc", font_size=20),
            Text("1 | 001 | 1", font_size=18, color=c1),
            Text("2 | 010 | 2", font_size=18, color=c2),
            Text("3 | 011 | 1", font_size=18, color=c1)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        # Assets
        disk1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg", color=c1)
        disk2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg", color=c2)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.place_in_area(table, 'B4', 'E6', scale_factor=0.8)
        self.play(FadeIn(table))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Logic representation
        bit0 = Text("Bit 0", color=c1, font_size=20)
        bit1 = Text("Bit 1", color=c2, font_size=20)
        
        self.place_at_grid(disk1, 'B3', scale_factor=0.7)
        self.place_at_grid(bit0, 'B5', scale_factor=0.7)
        
        line1 = Line(disk1.get_right(), bit0.get_left(), color=c3)
        self.play(FadeIn(disk1), FadeIn(bit0), Create(line1))
        
        self.place_at_grid(disk2, 'C3', scale_factor=0.7)
        self.place_at_grid(bit1, 'C5', scale_factor=0.7)
        
        line2 = Line(disk2.get_right(), bit1.get_left(), color=c3)
        self.play(FadeIn(disk2), FadeIn(bit1), Create(line2))
        
        self.wait(2)
