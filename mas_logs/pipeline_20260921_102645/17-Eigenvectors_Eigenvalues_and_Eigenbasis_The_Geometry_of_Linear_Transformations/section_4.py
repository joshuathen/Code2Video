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
        self.setup_layout("Real-world Application: Vibration & Stability", [
            "Eigenvalues represent system vibration frequencies.",
            "Eigenvectors define the bridge's vibration shape.",
            "Resonance occurs at specific eigen-frequencies."
        ])
        
        # Animations
        # Load assets
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        sine_wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-2, 2], color=WHITE)
        eigen_vector = Vector(UP, color="#FF00FF")
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg]
        self.place_in_area(sine_wave, 'A1', 'B6', scale_factor=0.6)
        self.play(Create(sine_wave))
        self.lecture[0].set_color("#FFFF00") # Light yellow
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Overlay eigen-mode vectors at resonance in magenta (#FF00FF)
        self.place_in_area(bridge, 'C1', 'D6', scale_factor=0.6)
        eigen_vector.next_to(bridge, UP)
        self.play(FadeIn(bridge), GrowArrow(eigen_vector))
        self.lecture[1].set_color("#0000FF") # Light blue
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display 'Stable/Unstable' label as eigenvalues change in red (#FF4500).
        label = Text("Unstable", color="#FF4500", font_size=24)
        self.place_at_grid(label, 'D3', scale_factor=0.5)
        self.play(Write(label))
        self.lecture[2].set_color("#00FF00") # Light green
        self.wait(2)
