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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Imagine a globe's surface.", "Any two opposite points exist.", "Temperatures at these points match.", "Antipodal points share conditions.", "This is Borsuk-Ulam's insight."]
        self.setup_layout("Intuitive Hook: Antipodal Points", lecture_lines)
        
        # Load asset
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        self.place_in_area(globe, 'B3', 'E6', scale_factor=0.85)
        
        # Define antipodal points (fixed relative to globe image)
        # Assuming globe is centered in its area
        p1 = Dot(color="#FF5733", radius=0.1)
        p2 = Dot(color="#33FF57", radius=0.1)
        
        # Place points on the "globe"
        p1.move_to(globe.get_center() + RIGHT * 0.5 + UP * 0.3)
        p2.move_to(globe.get_center() + LEFT * 0.5 + DOWN * 0.3)
        
        # Pulse animation
        def pulse(mob, dt):
            # Scale change not allowed, using radius updater
            mob.radius = 0.1 + 0.05 * np.sin(self.time * 5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(globe), run_time=1)
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(p1), FadeIn(p2), run_time=1)
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(p1), Indicate(p2), run_time=1)
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        # Using a workaround for pulsing without scaling the mob directly or always_redraw if possible
        # For simplicity and perf, just indicate them again or use a simpler visual cue
        self.play(Rotating(globe, radians=PI/2, about_point=globe.get_center()), run_time=2)
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(p1), FadeOut(p2), FadeOut(globe), run_time=1)
        self.lecture[4].set_color("#FFD700")
        self.wait(1)
