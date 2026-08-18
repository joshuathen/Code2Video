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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application: Decibel Scale", [
            "Sound intensity spans huge ranges.", 
            "Log scales compress this range.", 
            "Decibels simplify massive power numbers."
        ])
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"L = 10 \cdot \log_{10}\left(\frac{I}{I_0}\right)", color=WHITE)
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg", color=WHITE)
        
        # Using place_in_area as recommended by issue 31
        self.place_in_area(formula, 'B4', 'C6', scale_factor=1.0)
        self.place_at_grid(mic, 'B3', scale_factor=0.6)
        
        self.play(Write(formula), FadeIn(mic))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Using place_at_grid('D3') as recommended by issue 33
        step_highlight = VGroup(
            Circle(radius=0.2, color="#FF00FF"),
            Circle(radius=0.4, color="#FF00FF"),
            Circle(radius=0.8, color="#FF00FF")
        ).arrange(RIGHT)
        
        self.place_at_grid(step_highlight, 'D3', scale_factor=0.8)
        self.play(Create(step_highlight))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        intensity_val = Text("10^10", color="#FFFF00")
        decibel_val = Text("100 dB", color="#00FF00")
        ear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ear.svg", color=WHITE)
        
        # Using place_at_grid('E2') as recommended by issue 32
        self.place_at_grid(intensity_val, 'E2', scale_factor=0.8)
        self.place_at_grid(decibel_val, 'E5', scale_factor=0.8)
        self.place_at_grid(ear, 'F4', scale_factor=0.6)
        
        self.play(FadeIn(intensity_val), FadeIn(decibel_val), FadeIn(ear))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
