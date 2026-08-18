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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Constrained Towers of Hanoi", [
            "Disks move only between adjacent pegs in this variation.",
            "Each move state maps uniquely to a ternary string.",
            "Three disks create twenty-seven possible state configurations."
        ])
        
        # Load Assets
        peg_img = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        disk_img = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # Setup Pegs
        peg_a = SVGMobject(peg_img).set_color(WHITE)
        peg_b = SVGMobject(peg_img).set_color(WHITE)
        peg_c = SVGMobject(peg_img).set_color(WHITE)
        
        pegs = VGroup(peg_a, peg_b, peg_c).arrange(RIGHT, buff=1.0)
        self.place_in_area(pegs, 'B3', 'D5', scale_factor=0.8)
        
        # Setup Disks
        disk1 = SVGMobject(disk_img).set_color(BLUE)
        disk2 = SVGMobject(disk_img).set_color(GREEN)
        disk3 = SVGMobject(disk_img).set_color(YELLOW)
        
        disks = VGroup(disk3, disk2, disk1).arrange(DOWN, buff=0.1).next_to(peg_a, DOWN, buff=-0.5)
        
        self.add(pegs, disks)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEEB"))
        
        # Animate moving disk from peg A to B
        self.play(disk1.animate.move_to(peg_b.get_center() + UP*0.5), path_arc=PI/2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#98FB98"))
        
        # Highlight restriction (illegal move: A to C)
        illegal_arrow = Arrow(peg_a.get_top(), peg_c.get_top(), color="#FF6347")
        self.place_at_grid(illegal_arrow, 'C4', scale_factor=0.7)
        cross = Cross(illegal_arrow, color="#FF6347")
        self.play(Create(illegal_arrow), Create(cross))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Text label for 27 states
        states_text = Text("States: 3^3 = 27", font_size=24, color=WHITE)
        self.place_at_grid(states_text, 'E4', scale_factor=0.9)
        self.play(Write(states_text))
        
        self.wait(2)
