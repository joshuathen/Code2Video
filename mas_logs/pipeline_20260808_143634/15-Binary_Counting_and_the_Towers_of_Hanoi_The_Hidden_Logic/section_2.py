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
        self.setup_layout("The Towers of Hanoi Challenge", ["Move disks between three rods.", "Never place larger over smaller.", "Solve for N disks."])
        
        # Use SVG assets as requested
        rod_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg"
        disk_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        rod1 = SVGMobject(rod_svg, color="#A52A2A")
        rod2 = SVGMobject(rod_svg, color="#A52A2A")
        rod3 = SVGMobject(rod_svg, color="#A52A2A")
        rods = VGroup(rod1, rod2, rod3).arrange(RIGHT, buff=1.0)
        self.place_in_area(rods, 'C2', 'E6', scale_factor=0.9)
        self.add(rods)
        
        disks = VGroup(*[SVGMobject(disk_svg, color="#A52A2A").scale(0.5) for _ in range(3)])
        # Stack disks on rod1 (rods[0])
        for i, disk in enumerate(disks):
            disk.move_to(rods[0].get_bottom() + UP * (0.1 + i * 0.3))
            self.add(disk)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        goal_text = Text("Goal: Move all to peg C", color=WHITE, font_size=24)
        self.place_at_grid(goal_text, 'D4', scale_factor=0.9)
        self.play(Write(goal_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        
        # Flash invalid move: larger on smaller
        disk_to_move = SVGMobject(disk_svg, color="#A52A2A").scale(0.3)
        disk_to_move.move_to(rods[1].get_bottom() + UP * 0.5)
        
        flash = Rectangle(width=0.6, height=0.3, color=RED, fill_opacity=0.5)
        flash.move_to(disk_to_move.get_center())
        
        self.play(FadeIn(disk_to_move))
        self.play(FadeIn(flash), run_time=0.5)
        self.play(FadeOut(flash), FadeOut(disk_to_move), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
