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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Mechanism: Snell’s Law", [
            "Refractive index defines how light slows.",
            "Snell's law relates angles and indices.",
            "Notice the normal line and angles.",
            "Light bends when moving between media.",
            "Adjust the angle to see refraction shift."
        ])
        
        # Asset Loading (Placeholders)
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        snells_law = MathTex(r"n_1 \sin(\theta_1) = n_2 \sin(\theta_2)", color=WHITE)
        self.place_in_area(snells_law, 'B2', 'B5', scale_factor=0.8)
        self.place_at_grid(prism, 'B6', scale_factor=0.3)
        self.play(Write(snells_law), FadeIn(prism))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Setup Diagram elements
        boundary = Line(self.grid["D1"], self.grid["D6"], color=WHITE)
        normal = DashedLine(self.grid["B4"], self.grid["F4"], color=GRAY)
        
        # Incident Ray
        incident_ray = Line(self.grid["B2"], self.grid["D4"], color=WHITE)
        incident_angle_label = Text(r"$\theta_1$", color="#FFFF00", font_size=24)
        self.place_at_grid(incident_angle_label, 'C3', scale_factor=0.9)
        self.place_at_grid(laser, 'B1', scale_factor=0.3)
        
        self.play(Create(boundary), Create(normal), Create(incident_ray), Write(incident_angle_label), FadeIn(laser))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Refracted Ray
        refracted_ray = Line(self.grid["D4"], self.grid["F5"], color=WHITE)
        refracted_angle_label = Text(r"$\theta_2$", color="#00FF00", font_size=24)
        self.place_at_grid(refracted_angle_label, 'D4', scale_factor=0.9)
        
        self.play(Create(refracted_ray), Write(refracted_angle_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#AAAAAA")
        self.play(Indicate(snells_law))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF8800")
        self.place_at_grid(water, 'E6', scale_factor=0.3)
        self.play(FadeIn(water))
        
        # Simple dynamic simulation
        theta1 = ValueTracker(PI/4)
        
        # Update logic
        def update_ray(m):
            angle = theta1.get_value()
            m.put_start_and_end_on(
                self.grid["D4"] + np.array([-np.sin(angle), np.cos(angle), 0]) * 1.5,
                self.grid["D4"]
            )
            
        incident_ray.add_updater(update_ray)
        self.play(theta1.animate.set_value(PI/3), run_time=2)
        incident_ray.remove_updater(update_ray)
        self.wait(1)
