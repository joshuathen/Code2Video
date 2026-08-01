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
        # Data
        title_text = "Real-World Application: Quality Control"
        lecture_lines = [
            "Binomial logic helps factories predict part defects.",
            "We calculate the likelihood of exceeding failure limits.",
            "Statistics ensures quality control in mass production."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        c_white = "#FFFFFF"
        c_blue = "#ADD8E6"
        c_red = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Display a robot arm icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg] 
        # and a conveyor belt [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/con.svg] 
        # of gadgets in white #FFFFFF.
        
        robot_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color(c_white)
        conveyor_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/con.svg").set_color(c_white)
        
        # Group them together
        factory_viz = VGroup(conveyor_svg, robot_svg).arrange(RIGHT, buff=0.5)
        
        # [Fix Issue 39] Improved grid utilization: factory_viz at A1 to C4
        self.place_in_area(factory_viz, "A1", "C4", scale_factor=0.8)
        
        self.lecture[0].set_color(c_white)
        self.play(Create(factory_viz), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the parameters 'n=50, p=0.02' next to the conveyor belt in light blue #ADD8E6.
        params = MathTex("n=50, \\ p=0.02", color=c_blue, font_size=36)
        
        # [Fix Issue 37] Fix vertical overcrowding: parameters at B5, scale_factor=0.9
        self.place_at_grid(params, "B5", scale_factor=0.9)
        
        self.lecture[1].set_color(c_blue)
        self.play(Write(params))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the area of a distribution graph where X > 3 in red #FF0000 to represent 'Defect Risk'.
        
        # Binomial PMF values for n=50, p=0.02
        pmf_data = [0.364, 0.371, 0.185, 0.061, 0.015, 0.003, 0.001]
        
        axes = Axes(
            x_range=[0, 7, 1],
            y_range=[0, 0.5, 0.1],
            x_length=4,
            y_length=2.5,
            axis_config={"include_tip": False},
        )
        
        bars = VGroup()
        for k, val in enumerate(pmf_data):
            # bar_height calculation relative to axes
            # axes.c2p returns scene coordinates. 
            # We want a bar from y=0 to y=val.
            p1 = axes.c2p(k, 0)
            p2 = axes.c2p(k, val)
            
            # Using Rectangle for simplicity as in previous code, but adjusting height correctly
            # Height in scene units:
            height = p2[1] - p1[1]
            bar = Rectangle(
                width=0.3,
                height=height,
                fill_opacity=0.8,
                stroke_width=1
            )
            bar.move_to(axes.c2p(k, val/2))
            
            if k > 3:
                bar.set_color(c_red)
                bar.set_fill(c_red)
            else:
                bar.set_color(c_white)
                bar.set_fill(c_white)
            bars.add(bar)
            
        risk_label = Text("Defect Risk (X > 3)", color=c_red, font_size=24)
        risk_label.next_to(axes, UP, buff=0.1)
        
        graph_group = VGroup(axes, bars, risk_label)
        
        # [Fix Issue 38] Fix visual collision: graph_group area D2 to F6, scale_factor=0.8
        self.place_in_area(graph_group, "D2", "F6", scale_factor=0.8)
        
        self.lecture[2].set_color(c_red)
        self.play(
            Create(axes),
            Create(bars),
            Write(risk_label),
            run_time=2
        )
        self.wait(2)
