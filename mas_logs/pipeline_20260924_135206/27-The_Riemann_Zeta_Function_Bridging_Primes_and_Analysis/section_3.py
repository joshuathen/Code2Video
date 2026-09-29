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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Analytic Continuation: Mapping the Invisible", [
            "Standard summation only works when s is large.",
            "We use analytic continuation to map the plane.",
            "Think of extending a road beyond the horizon."
        ])
        
        # --- Visualization Elements ---
        # Domain boundary for Re(s) > 1
        domain = Rectangle(width=2.5, height=4, color="#00FFFF", fill_opacity=0.3)
        # Using columns 4-6 to isolate from lecture text
        self.place_in_area(domain, 'A4', 'C6', scale_factor=0.7)
        
        # Asset images
        horizon_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/horizon.svg")
        road_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/road.svg")
        
        # \"Hidden landscape\" mountain surface
        landscape = Surface(
            lambda u, v: np.array([u, v, np.sin(3*u) * np.cos(3*v)]),
            u_range=[-1.5, 1.5], v_range=[-1.5, 1.5],
            resolution=(15, 15)
        ).set_fill(color="#FFFFFF", opacity=0.5).set_stroke(color="#FFFFFF", width=0.5)
        
        # Pulsing circle connection point
        connection_point = Circle(radius=0.2, color="#800080", fill_opacity=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(horizon_icon, 'A5', scale_factor=0.5)
        self.play(FadeIn(horizon_icon), Create(domain))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_in_area(landscape, 'D4', 'F6', scale_factor=0.5)
        self.place_at_grid(road_icon, 'D5', scale_factor=0.5)
        self.play(Create(landscape), FadeIn(road_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#800080"))
        self.place_at_grid(connection_point, 'C5', scale_factor=0.5)
        self.play(FadeIn(connection_point), run_time=0.5)
        self.play(Indicate(connection_point), run_time=1.5)
        self.wait(2)
