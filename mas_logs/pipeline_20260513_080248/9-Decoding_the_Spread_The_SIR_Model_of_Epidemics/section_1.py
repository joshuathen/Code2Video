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
        # Setup the basic layout
        lecture_lines = [
            'Infectious diseases seem chaotic and unpredictable.', 
            'Mathematics reveals hidden patterns in how they spread.', 
            'Today, we decode the math behind the sneeze.'
        ]
        self.setup_layout("Introduction: The Math Behind the Sneeze", lecture_lines)
        
        # Constants for simulation
        BLUE_CLR = "#3498db"
        RED_CLR = "#e74c3c"
        HIGHLIGHT_CLR = "#f1c40f"
        
        # Grid boundaries for simulation (B2 to F6)
        # B2: (1.5, 1.2), F6: (5.5, -2.8). Span is 4.0.
        # With scale factor 0.9, effective width/height is 3.6.
        box_center = (self.grid['B2'] + self.grid['F6']) / 2
        side = 4.0 * 0.9
        x_min, x_max = box_center[0] - side/2, box_center[0] + side/2
        y_min, y_max = box_center[1] - side/2, box_center[1] + side/2

        # Simulation space visualization
        box = Rectangle(width=4.0, height=4.0, color=WHITE, stroke_opacity=0.3)
        self.place_in_area(box, 'B2', 'F6', scale_factor=0.9)
        self.add(box)

        # === Animation for Lecture Line 1 ===
        # Initialize 20 blue circles moving randomly
        self.lecture[0].set_color(HIGHLIGHT_CLR)
        
        dots = VGroup()
        for _ in range(20):
            d = Dot(radius=0.1, color=BLUE_CLR)
            d.move_to([
                np.random.uniform(x_min + 0.1, x_max - 0.1),
                np.random.uniform(y_min + 0.1, y_max - 0.1),
                0
            ])
            d.vel = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                0
            ]) * 0.8
            d.is_infected = False
            dots.add(d)
        
        def update_dots(mob, dt):
            for d in mob:
                d.shift(d.vel * dt)
                # Bounce mechanics
                if d.get_x() <= x_min or d.get_x() >= x_max:
                    d.vel[0] *= -1
                    # Push back inside bounds
                    d.set_x(np.clip(d.get_x(), x_min, x_max))
                if d.get_y() <= y_min or d.get_y() >= y_max:
                    d.vel[1] *= -1
                    d.set_y(np.clip(d.get_y(), y_min, y_max))
            
            # Infection logic: check for collisions
            infected = [d for d in mob if d.is_infected]
            susceptible = [d for d in mob if not d.is_infected]
            for s in susceptible:
                for i in infected:
                    if np.linalg.norm(s.get_center() - i.get_center()) < 0.22:
                        s.is_infected = True
                        s.set_color(RED_CLR)
                        break

        self.add(dots)
        dots.add_updater(update_dots)
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_CLR)

        # Find center-most dot to infect
        center_pt = box_center
        patient_zero = min(dots, key=lambda d: np.linalg.norm(d.get_center() - center_pt))
        
        self.play(
            Indicate(patient_zero, color=RED_CLR, scale_factor=2),
            patient_zero.animate.set_color(RED_CLR),
            run_time=1
        )
        patient_zero.is_infected = True
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_CLR)
        
        # Continue simulation to show spread
        self.wait(5)
        
        # Finish
        dots.remove_updater(update_dots)
        self.wait(1)
