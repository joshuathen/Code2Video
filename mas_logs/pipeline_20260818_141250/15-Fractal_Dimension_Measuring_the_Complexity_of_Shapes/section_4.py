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
        lecture_lines = [
            "Fractals appear frequently throughout the natural world.",
            "Bronchial trees maximize surface area for gas exchange.",
            "High fractal dimension indicates complex, dense structures."
        ]
        self.setup_layout("Visualizing Complexity in Nature and Technology", lecture_lines)
        
        # Load asset
        lung_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lung.svg"
        lung = SVGMobject(lung_path)
        
        # === Animation for Lecture Line 1 ===
        # Draw a simple bronchial tree structure appearing inside the lung SVG
        self.place_at_grid(lung, 'B4', scale_factor=1.5)
        self.play(Create(lung), run_time=2)
        lung.set_color("#00FFFF")
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight the branches filling space within the 3D volume
        self.play(lung.animate.set_color("#FFCC00"), run_time=1.5)
        self.lecture[1].set_color("#FFCC00")

        # === Animation for Lecture Line 3 ===
        # Show the surface area increasing to visualize the 'D' value
        d_val = MathTex(r"D \approx 2.7", color="#FFFFFF")
        self.place_at_grid(d_val, 'E4', scale_factor=1.2)
        
        self.play(FadeIn(d_val), lung.animate.set_color("#FFFFFF"), run_time=1.5)
        self.lecture[2].set_color("#FFFFFF")
        
        self.wait(2)
