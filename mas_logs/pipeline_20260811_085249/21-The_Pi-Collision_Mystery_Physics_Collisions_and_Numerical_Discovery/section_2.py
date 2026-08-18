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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We map velocities onto a two-dimensional plane.",
            "Each velocity pair defines a point in space.",
            "Collisions act as reflections against a circular arc.",
            "The point bounces within this curved boundary.",
            "Geometry reveals the path of our physical system."
        ]
        self.setup_layout("Visualizing the Mapping: Momentum to Geometry", lecture_lines)
        
        # --- Create Animation Elements ---
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True}).scale(0.6)
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg]
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")
        arc = Arc(radius=1.2, start_angle=0, angle=PI, color=BLUE)
        
        point_label = Text("Point", font_size=16)
        arc_label = Text("Boundary", font_size=16)
        
        # --- Animations ---
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.6)
        self.play(Create(axes))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(particle, 'E3', scale_factor=0.3)
        self.play(FadeIn(particle))
        self.place_at_grid(point_label, 'E4', scale_factor=0.5)
        self.play(Write(point_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.place_at_grid(arc, 'E3', scale_factor=0.6)
        self.play(Create(arc))
        self.place_at_grid(arc_label, 'E2', scale_factor=0.5)
        self.play(Write(arc_label))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        # Particle movement
        self.play(particle.animate.move_to(self.grid['E2']), run_time=1)
        self.play(particle.animate.move_to(self.grid['D3']), run_time=1)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.wait(1)
