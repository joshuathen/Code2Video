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
        self.setup_layout("Mapping Collisions to Geometry", [
            "Unwrap 1D collisions into 2D geometric paths.",
            "Represent the state as a single ray.",
            "Bounces map to reflections within a wedge."
        ])
        
        # Load Assets
        ray_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ray.svg", color="#FF5733")
        wedge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wedge.svg", color="#33FF57")
        
        # Positioning using updated requirements
        self.place_at_grid(ray_icon, 'C2', scale_factor=0.8)
        self.place_at_grid(wedge_icon, 'C3', scale_factor=0.8)
        
        # Keep original logic as placeholder for secondary visuals
        ray = Line(start=np.array([0, 0, 0]), end=np.array([0.5, 0.5, 0]), color="#FF5733")
        dashed_line = DashedLine(start=np.array([0, 0, 0]), end=np.array([0, 0.5, 0]), color="#33FF57")
        
        self.place_at_grid(ray, 'D2', scale_factor=0.7)
        self.place_at_grid(dashed_line, 'D3', scale_factor=0.7)
        
        angle_text = Text("θ = 45°", font_size=24, color=WHITE)
        self.place_at_grid(angle_text, 'E2', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(ray_icon), FadeIn(wedge_icon))
        self.play(self.lecture[0].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 2 ===
        self.play(Create(ray), Create(dashed_line))
        self.play(self.lecture[1].animate.set_color("#33FF57"))

        # === Animation for Lecture Line 3 ===
        self.play(Write(angle_text))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
