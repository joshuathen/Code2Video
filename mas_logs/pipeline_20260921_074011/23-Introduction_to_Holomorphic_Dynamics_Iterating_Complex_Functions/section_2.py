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
            "Iteration is applying the function repeatedly.",
            "Dynamics study long-term behavior of sequences.",
            "Points either converge, wander, or cycle."
        ]
        self.setup_layout("The Concept of Iteration", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg]
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        self.place_at_grid(compass, "B4", scale_factor=0.6)
        
        z0 = Dot(color=WHITE)
        self.place_at_grid(z0, "B4") # Placed in B4 to avoid crowding
        
        self.play(FadeIn(compass), FadeIn(z0))
        label_z0 = Text("z_0", font_size=24, color=WHITE).next_to(z0, UP)
        self.play(Write(label_z0))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        formula = MathTex("z_{n+1} = z_n^2 + c", color=YELLOW)
        self.place_in_area(formula, "C3", "D5", scale_factor=0.9)
        self.play(Write(formula))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/orbit.svg]
        orbit_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orbit.svg", color=GREEN)
        
        # Simulate orbit path
        orbit_path = VGroup()
        orbit_path.add(orbit_icon)
        
        # Place orbit_path in B4-E6 area
        self.place_in_area(orbit_path, "B4", "E6", scale_factor=0.7)
        
        self.play(Create(orbit_path), run_time=2)
        orbit_label = Text("Orbit", font_size=20, color=GREEN).next_to(orbit_path, RIGHT)
        self.play(Write(orbit_label))
        
        self.wait(2)
