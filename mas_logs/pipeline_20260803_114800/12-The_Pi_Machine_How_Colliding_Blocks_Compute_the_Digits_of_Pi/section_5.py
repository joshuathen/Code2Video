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
        lecture_lines = [
            "The mass ratio determines the angle of each jump.",
            "Heavier masses result in much smaller angular steps.",
            "Total collisions represent a trip halfway around the circle.",
            "We divide Pi radians by this tiny collision angle.",
            "This explains why the digits of Pi emerge."
        ]
        self.setup_layout("The Final Reveal: Why Pi?", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        
        # Zoom into the circle to show the angle theta between points.
        center = self.grid["D3"]
        zoom_arc = Arc(radius=2.5, start_angle=-PI/6, angle=PI/3, color=BLUE)
        # Using a temporary container to use place_at_grid or move_to the grid point
        zoom_arc.move_to(center)
        
        p1 = zoom_arc.point_from_proportion(0.3)
        p2 = zoom_arc.point_from_proportion(0.7)
        
        line1 = Line(center, p1, color=GREY_A)
        line2 = Line(center, p2, color=GREY_A)
        
        angle_theta = Angle(line1, line2, radius=0.6, color=WHITE)
        theta_label = MathTex(r"\theta", color=WHITE)
        self.place_at_grid(theta_label, "D4", scale_factor=0.8)
        
        self.play(Create(zoom_arc), Create(line1), Create(line2))
        self.play(Create(angle_theta), Write(theta_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        # Label the angle as arctan(sqrt(m/M)) in #00FF00, using [Asset: ...]
        theta_formula = MathTex(r"\theta = \arctan\left(\sqrt{\frac{m}{M}}\right)", color="#00FF00")
        self.place_in_area(theta_formula, "B4", "B5", scale_factor=0.8)
        
        mass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mass.svg", color="#00FF00")
        self.place_at_grid(mass_icon, "A4", scale_factor=0.5)
        
        self.play(Write(theta_formula), FadeIn(mass_icon))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(PURPLE_A)
        
        # Show the total arc of the semicircle representing pi radians.
        self.play(
            FadeOut(zoom_arc), FadeOut(line1), FadeOut(line2), 
            FadeOut(angle_theta), FadeOut(theta_label), FadeOut(theta_formula),
            FadeOut(mass_icon)
        )
        
        full_arc = Arc(radius=1.8, start_angle=0, angle=PI, color=PURPLE_A)
        self.place_in_area(full_arc, "C3", "E5") # Centralizing the arc
        
        arc_center = (self.grid["C3"] + self.grid["E5"]) / 2
        
        pi_label = MathTex(r"\pi \text{ radians}", color=PURPLE_A)
        self.place_at_grid(pi_label, "B3", scale_factor=0.9)
        
        self.play(Create(full_arc))
        self.play(Write(pi_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GOLD)
        
        # Animate the path of chords filling the pi-radian arc.
        num_steps = 15
        chords = VGroup()
        for i in range(num_steps):
            start_ang = i * (PI / num_steps)
            end_ang = (i + 1) * (PI / num_steps)
            start_pt = arc_center + 1.8 * np.array([np.cos(start_ang), np.sin(start_ang), 0])
            end_pt = arc_center + 1.8 * np.array([np.cos(end_ang), np.sin(end_ang), 0])
            chords.add(Line(start_pt, end_pt, color=GOLD, stroke_width=3))
            
        self.play(Create(chords, lag_ratio=0.2), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        # Display 'Total collisions = Pi / theta' in #FFFF00.
        final_formula = MathTex(r"\text{Total collisions} \approx \frac{\pi}{\theta}", color="#FFFF00")
        self.place_in_area(final_formula, "F3", "F5", scale_factor=0.9)
        
        self.play(Write(final_formula))
        self.wait(3)
