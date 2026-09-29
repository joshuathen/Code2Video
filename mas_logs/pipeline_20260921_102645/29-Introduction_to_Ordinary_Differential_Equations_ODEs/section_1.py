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
        self.setup_layout("The Hook: Dynamics in Motion", [
            "We observe change all around us.",
            "Often, we only know how things change.",
            "ODEs help us predict future states."
        ])
        
        # Assets
        planet_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg"
        pendulum_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg"
        
        # === Animation for Lecture Line 1 ===
        # Fix for issue 22: Title placement
        title_obj = Text("Dynamics in Motion", font_size=32, color=WHITE)
        self.place_in_area(title_obj, "A2", "A5", scale_factor=0.9)
        self.play(FadeIn(title_obj))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fix for issue 21: Particle placement
        particles = VGroup(*[Dot(radius=0.04, color="#FF00FF") for _ in range(20)])
        planet = SVGMobject(planet_path).set_color(WHITE)
        self.place_at_grid(planet, "D3", scale_factor=0.5)
        
        dynamics_group = VGroup(particles, planet)
        self.place_in_area(dynamics_group, "D2", "F5", scale_factor=0.7)
        
        self.add(dynamics_group)
        self.play(
            FadeOut(title_obj),
            *[p.animate.shift(np.random.uniform(-0.5, 0.5, 3)) for p in particles],
            self.lecture[1].animate.set_color("#FF00FF"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        pendulum = SVGMobject(pendulum_path).set_color(WHITE)
        self.place_at_grid(pendulum, "C4", scale_factor=0.5)
        
        flow = VGroup(*[Line(start=self.grid["C2"], end=self.grid["D5"], color="#00FFFF") for _ in range(3)])
        flow.arrange(RIGHT, buff=0.2)
        
        scene_elements = VGroup(dynamics_group, flow, pendulum)
        self.place_in_area(scene_elements, "D2", "F5", scale_factor=0.7)
        
        self.play(
            Create(flow),
            FadeIn(pendulum),
            self.lecture[2].animate.set_color("#00FFFF"),
            run_time=2
        )
        self.wait(2)
