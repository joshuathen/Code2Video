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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Practical Application: The Sugar Concentration Sensor", 
                          ["Rotation indicates sugar concentration.", 
                           "We measure this angle precisely.", 
                           "This enables non-invasive testing."])
        
        # 1. Assets setup
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        beaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beaker.svg")
        detector = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detector.svg")
        
        # Positions based on feedback
        self.place_at_grid(laser, 'C2', scale_factor=0.6)
        self.place_at_grid(beaker, 'C3', scale_factor=0.9)
        self.place_at_grid(detector, 'C4', scale_factor=0.8)
        
        self.play(FadeIn(laser), FadeIn(beaker), FadeIn(detector))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture_mobjects[0].animate.set_color("#FFD700"))
        
        # 2. Label
        concentration_label = Text("Conc: 0%", font_size=24, color="#00FF00")
        self.place_at_grid(concentration_label, 'B5', scale_factor=0.8)
        self.play(Write(concentration_label))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture_mobjects[1].animate.set_color("#FFD700"))
        
        # 3. Visualize
        beam = Line(start=laser.get_right(), end=detector.get_left(), color=YELLOW)
        self.add(beam)
        
        angle_tracker = ValueTracker(0)
        beam.add_updater(lambda m: m.rotate(angle_tracker.get_value(), about_point=laser.get_right()))
        
        self.play(angle_tracker.animate.set_value(PI/6), run_time=2)
        
        # Keep label updated efficiently
        def update_label(m):
            new_text = Text(f"Conc: {int(angle_tracker.get_value()*20)}%", font_size=24, color="#00FF00")
            new_text.move_to(concentration_label.get_center())
            m.become(new_text)

        concentration_label.add_updater(update_label)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture_mobjects[2].animate.set_color("#FFD700"))
        self.wait(2)
