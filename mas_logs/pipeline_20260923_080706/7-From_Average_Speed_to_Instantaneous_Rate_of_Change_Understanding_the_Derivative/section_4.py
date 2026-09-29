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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: Prediction and Dynamics", [
            "Derivatives show change at an exact instant.",
            "They are essential for predicting dynamic motion.",
            "Use them to optimize paths like rocket launches."
        ])
        
        # Define Path and Objects
        curve = FunctionGraph(lambda t: 0.1 * (t**3 - 3*t), x_range=[-2, 2], color=GRAY)
        
        # Asset integration
        rocket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg", color=WHITE)
        
        velocity_vector = Arrow(start=ORIGIN, end=RIGHT*0.5, color="#00FFFF")
        tangent_line = Line(start=LEFT*1, end=RIGHT*1, color="#FFA500")

        # Tracker for position on curve
        t = ValueTracker(0)
        
        def update_objects(m):
            prop = t.get_value()
            pos = curve.point_from_proportion(prop)
            rocket.move_to(pos)
            
            # Derivative calculation (approximate slope)
            delta = 0.01
            p1 = curve.point_from_proportion(max(0, prop - delta))
            p2 = curve.point_from_proportion(min(1, prop + delta))
            tangent_dir = p2 - p1
            tangent_dir /= np.linalg.norm(tangent_dir)
            
            velocity_vector.put_start_and_end_on(pos, pos + tangent_dir * 0.8)
            tangent_line.put_start_and_end_on(pos - tangent_dir * 0.5, pos + tangent_dir * 0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Apply Kritik 1
        self.place_in_area(curve, 'B3', 'E5', scale_factor=0.9)
        self.add(curve)
        self.play(Create(curve))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        # Apply Kritik 2
        self.place_at_grid(rocket, 'C3', scale_factor=0.6)
        
        rocket.add_updater(update_objects)
        velocity_vector.add_updater(update_objects)
        self.add(rocket, velocity_vector)
        self.play(t.animate.set_value(1), run_time=3, rate_func=linear)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFA500")
        # Apply Kritik 3
        self.place_in_area(curve, 'C2', 'D5', scale_factor=0.75)
        
        tangent_line.add_updater(update_objects)
        self.add(tangent_line)
        self.play(t.animate.set_value(0), run_time=3, rate_func=linear)
        
        self.wait(1)
