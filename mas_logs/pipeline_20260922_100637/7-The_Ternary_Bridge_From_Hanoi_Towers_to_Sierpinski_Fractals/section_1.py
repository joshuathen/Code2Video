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
            "Towers of Hanoi map perfectly to base-three logic.",
            "Each move transition corresponds to a ternary digit change.",
            "Visualize disk positions as ternary numbers."
        ]
        self.setup_layout("The Hidden Base-3 Language", lecture_lines)
        
        # Assets
        # Using SVG file as referenced in storyboard: /scratch/pawsey1357/jthen/Code2Video/assets/icon/discs.svg
        # Note: Since I don't have direct disk file access, I'll use SVGMobject or a proxy.
        # Assuming existence:
        disks_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/discs.svg")
        
        # Construct Hanoi visualization (using basic shapes for animation)
        # 3 Pegs
        pegs = VGroup(*[Rectangle(height=1.5, width=0.15, color=GRAY, fill_opacity=1) for _ in range(3)])
        pegs.arrange(RIGHT, buff=1.0)
        
        # 3 Disks
        disks = VGroup(*[Rectangle(height=0.25, width=0.8 + i*0.4, color="#FFD700", fill_opacity=1) for i in range(3)])
        
        towers_group = VGroup(pegs, *disks)
        self.place_in_area(towers_group, 'C4', 'F6', scale_factor=0.6)
        
        # Set positions
        for i in range(3):
            disks[i].move_to(pegs[0].get_bottom() + UP * (0.125 + i * 0.25))

        self.add(towers_group)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        # Move bottom disk (index 2) to peg 1 (middle)
        self.play(disks[2].animate.move_to(pegs[1].get_bottom() + UP * 0.125))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Shift middle disk (index 1) to peg 2 (right)
        self.play(disks[1].animate.move_to(pegs[2].get_bottom() + UP * 0.125))
        self.wait(2)
