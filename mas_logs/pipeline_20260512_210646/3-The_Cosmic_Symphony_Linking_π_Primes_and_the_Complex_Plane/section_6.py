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
        # Setup layout with title and lecture lines
        lecture_lines = [
            'Pi, rotation, and primes weave a single mathematical fabric.', 
            'This symmetry secures the encryption of our digital world.', 
            'Geometry and numbers dance in one cosmic symphony.'
        ]
        self.setup_layout("Conclusion: The Universal Code", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Description: Display the symbols π, i, and p (primes) rotating slowly around a central point (#FFFFFF).
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Create symbols using Unicode
        pi_sym = Text("π", color="#FFFFFF", font_size=48)
        i_sym = Text("i", color="#FFFFFF", font_size=48)
        p_sym = Text("p", color="#FFFFFF", font_size=48)
        
        # Initial placement around a central area
        self.place_at_grid(pi_sym, "C3", scale_factor=1.3)
        self.place_at_grid(i_sym, "C5")
        self.place_at_grid(p_sym, "E4")
        
        # Central point visualization
        center_point_obj = Dot(color="#FFFFFF", radius=0.05)
        self.place_at_grid(center_point_obj, "D4")
        center_pos = center_point_obj.get_center()
        
        symbols_group = VGroup(pi_sym, i_sym, p_sym)
        self.play(FadeIn(symbols_group), FadeIn(center_point_obj))
        
        # Slow rotation around the central point
        self.play(
            Rotate(symbols_group, angle=TAU, about_point=center_pos, run_time=4, rate_func=linear)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Description: Transform these symbols into a digital padlock icon (#00FF00) representing modern encryption.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Construct Padlock
        padlock_body = RoundedRectangle(height=1.2, width=1.4, corner_radius=0.1, fill_opacity=1, color="#00FF00")
        # Shackle is an arc
        padlock_shackle = Arc(radius=0.5, start_angle=0, angle=PI, color="#00FF00", stroke_width=10)
        padlock_shackle.shift(UP * 0.6) # Relative shift within the local group construction
        
        padlock_icon = VGroup(padlock_body, padlock_shackle)
        self.place_in_area(padlock_icon, "C3", "E5", scale_factor=0.8)
        
        # Transform symbols into padlock
        self.play(
            ReplacementTransform(symbols_group, padlock_icon),
            FadeOut(center_point_obj),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Fade out everything except the text 'The Cosmic Symphony' in large, glowing letters (#FFD700).
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        symphony_text = Text("The Cosmic Symphony", color="#FFD700", font_size=42)
        # Position in a broad central area of the right side
        self.place_in_area(symphony_text, "C2", "D5", scale_factor=0.9)
        
        self.play(
            FadeOut(padlock_icon),
            FadeIn(symphony_text)
        )
        
        # Simulate 'glowing' pulse effect
        self.play(
            symphony_text.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(3)
