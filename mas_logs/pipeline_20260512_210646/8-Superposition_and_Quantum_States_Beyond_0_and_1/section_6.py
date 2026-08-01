from manim import *
import numpy as np

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with correct lecture lines
        lecture_lines = [
            'Vectors, superposition, and measurement define quantum logic.',
            'Quantum parallelism explores all solutions at once.',
            'This power fuels the future of advanced computing.'
        ]
        self.setup_layout("Summary and Real-World Application", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Stage: Summary_Flow
        # Animation: The terms 'Vector' -> 'Superposition' -> 'Measurement' appear in sequence in white (#FFFFFF).
        self.lecture[0].set_color(WHITE)
        
        txt_vector = Text("Vector", font_size=24, color="#FFFFFF")
        txt_super = Text("Superposition", font_size=24, color="#FFFFFF")
        txt_meas = Text("Measurement", font_size=24, color="#FFFFFF")
        
        # Grid Fixes from Issue 40 - Moving labels to Row A to avoid overlap
        self.place_at_grid(txt_vector, 'A1', scale_factor=0.8)
        self.place_at_grid(txt_super, 'A3', scale_factor=0.7)
        self.place_at_grid(txt_meas, 'A5', scale_factor=0.7)
        
        arrow1 = Arrow(self.grid["A1"], self.grid["A3"], buff=0.5, color="#FFFFFF", stroke_width=2)
        arrow2 = Arrow(self.grid["A3"], self.grid["A5"], buff=0.5, color="#FFFFFF", stroke_width=2)
        
        self.play(Write(txt_vector))
        self.play(GrowArrow(arrow1))
        self.play(Write(txt_super))
        self.play(GrowArrow(arrow2))
        self.play(Write(txt_meas))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Stage: Maze_Parallelism
        # Animation: A maze grid [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/maze.svg] 
        # appears where multiple paths are simultaneously highlighted in cyan (#00FFFF).
        self.play(
            FadeOut(txt_vector, txt_super, txt_meas, arrow1, arrow2),
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Asset Integration from Issue 26
        maze_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/maze.svg")
        maze_svg.set_color(WHITE)
        self.place_in_area(maze_svg, "B2", "E5", scale_factor=1.0)
        
        # Simultaneous paths in cyan, scaled relative to the maze SVG bounds
        mw = maze_svg.width
        mh = maze_svg.height
        dl_corner = maze_svg.get_corner(DL)
        ur_target = maze_svg.get_corner(UR)
        
        # Path 1: Top-Right biased
        p1 = VMobject(color="#00FFFF", stroke_width=4)
        p1.set_points_as_corners([
            dl_corner + UR*0.1,
            dl_corner + RIGHT*0.3*mw + UP*0.1,
            dl_corner + RIGHT*0.3*mw + UP*0.8*mh,
            ur_target - UR*0.1
        ])
        
        # Path 2: Central horizontal bypass
        p2 = VMobject(color="#00FFFF", stroke_width=4)
        p2.set_points_as_corners([
            dl_corner + UR*0.1,
            dl_corner + UP*0.5*mh + RIGHT*0.1,
            dl_corner + UP*0.5*mh + RIGHT*0.9*mw,
            ur_target - UR*0.1
        ])

        # Path 3: Bottom-Right biased
        p3 = VMobject(color="#00FFFF", stroke_width=4)
        p3.set_points_as_corners([
            dl_corner + UR*0.1,
            dl_corner + RIGHT*0.8*mw + UP*0.1,
            dl_corner + RIGHT*0.8*mw + UP*0.9*mh,
            ur_target - UR*0.1
        ])

        self.play(FadeIn(maze_svg))
        self.play(Create(p1), Create(p2), Create(p3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Stage: Quantum_Computer
        # Animation: A stylized quantum chip icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/chip.svg] 
        # in medium slate blue (#7B68EE) fades in to conclude the video.
        self.play(
            FadeOut(maze_svg, p1, p2, p3),
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#7B68EE")
        )
        
        # Asset Integration from Issue 26 and Sizing Fix from Issue 41
        chip_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/chip.svg")
        chip_svg.set_color("#7B68EE")
        self.place_in_area(chip_svg, 'B2', 'E5', scale_factor=1.0)
        
        self.play(FadeIn(chip_svg, scale=0.8))
        self.play(chip_svg.animate.set_fill(WHITE, opacity=0.3), run_time=0.5, rate_func=there_and_back)
        self.wait(2)
