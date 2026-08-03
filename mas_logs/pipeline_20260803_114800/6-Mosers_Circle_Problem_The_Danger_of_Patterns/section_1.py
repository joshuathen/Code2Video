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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup layout
        self.setup_layout("The Setup: Connecting the Dots", 
                          ["Place n points on a circle's circumference.", 
                           "Connect every point to every other point.", 
                           "How many regions do these lines create?"])
        
        # Colors
        CIRCLE_COLOR = "#FFFFFF"
        POINT_COLOR = "#FF00FF"
        LINE_COLOR = "#00FFFF"
        TEXT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "Place n points on a circle's circumference."
        self.lecture[0].set_color(YELLOW)
        
        # Issue 22: Use asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        # Issue 26: Move circle to C2 to F5
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg").set_color(CIRCLE_COLOR)
        self.place_in_area(circle, 'C2', 'F5', scale_factor=1.0)
        
        center = circle.get_center()
        # Since it's an SVG, we estimate the visual radius for point placement.
        # SVGMobjects have a height/width, we use half the height as radius.
        radius = circle.height / 2
        
        def get_pt(angle_deg):
            a = angle_deg * DEGREES
            return center + radius * np.array([np.cos(a), np.sin(a), 0])

        # Draw the circle and place two magenta points
        self.play(Create(circle))
        
        p1 = Dot(get_pt(30), color=POINT_COLOR)
        p2 = Dot(get_pt(210), color=POINT_COLOR)
        points = VGroup(p1, p2)
        
        self.play(FadeIn(points))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Connect every point to every other point."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Connect points with a cyan line (n=2, creates 2 regions)
        line1 = Line(p1.get_center(), p2.get_center(), color=LINE_COLOR)
        self.play(Create(line1))
        
        # Add a third point and connect to all others
        p3 = Dot(get_pt(120), color=POINT_COLOR)
        l3_1 = Line(p3.get_center(), p1.get_center(), color=LINE_COLOR)
        l3_2 = Line(p3.get_center(), p2.get_center(), color=LINE_COLOR)
        
        self.play(FadeIn(p3))
        self.play(Create(l3_1), Create(l3_2))
        points.add(p3)
        self.wait(1)

        # Add fourth point and connect, highlighting regions doubling
        p4 = Dot(get_pt(-60), color=POINT_COLOR)
        l4_1 = Line(p4.get_center(), p1.get_center(), color=LINE_COLOR)
        l4_2 = Line(p4.get_center(), p2.get_center(), color=LINE_COLOR)
        l4_3 = Line(p4.get_center(), p3.get_center(), color=LINE_COLOR)
        
        self.play(FadeIn(p4))
        self.play(Create(l4_1), Create(l4_2), Create(l4_3))
        points.add(p4)
        
        # Issue 25: Visual cue for doubling: Move doubling_text to area B3 to B5
        doubling_text = Text("2 -> 4 -> 8 regions", font_size=24, color=TEXT_COLOR)
        self.place_in_area(doubling_text, 'B3', 'B5', scale_factor=0.9)
        self.play(Write(doubling_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "How many regions do these lines create?"
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Label regions sequentially as they increase to 8 in #FFFF00
        # Positioning labels in grid cells that correspond to the regions within the circle area (C2-F5)
        region_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]
        region_positions = ["D3", "D4", "E3", "E4", "C3", "E2", "F4", "D5"]
        
        label_mobs = VGroup()
        for i, pos in enumerate(region_positions):
            lbl = Text(region_labels[i], font_size=24, color=TEXT_COLOR)
            self.place_at_grid(lbl, pos, scale_factor=0.8)
            label_mobs.add(lbl)
            self.play(FadeIn(lbl), run_time=0.4)
            
        self.wait(2)
