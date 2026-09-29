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
        self.setup_layout("Summary and Conclusion", [
            "Divergence tracks expansion and contraction.",
            "Curl detects the rotation of fields.",
            "These concepts underpin fundamental physical laws."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Represent Divergence (Expansion)
        star = Star(n=5, color=YELLOW, fill_opacity=0.5)
        self.place_in_area(star, 'A2', 'A3', scale_factor=0.6)
        div_label = Text("Divergence", font_size=20, color=YELLOW)
        self.place_at_grid(div_label, 'A4', scale_factor=0.7)
        
        self.play(FadeIn(star), Write(div_label))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Represent Curl (Rotation)
        cyclone = VGroup(
            Arc(radius=0.4, start_angle=0, angle=PI*1.5, color=BLUE),
            Arrow(start=ORIGIN, end=RIGHT*0.2, color=BLUE)
        )
        self.place_in_area(cyclone, 'C2', 'C3', scale_factor=0.6)
        curl_label = Text("Curl", font_size=20, color=BLUE)
        self.place_at_grid(curl_label, 'C4', scale_factor=0.7)
        
        self.play(FadeIn(cyclone), Write(curl_label))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        # Physics Laws icon/text
        physics_icon = MathTex(r"\nabla \cdot \mathbf{F} = \rho", color=WHITE)
        self.place_in_area(physics_icon, 'E2', 'E5', scale_factor=0.8)
        
        self.play(Write(physics_icon))
        self.lecture[2].set_color(WHITE)
        
        self.wait(2)
        self.play(FadeOut(self.title), FadeOut(self.lecture), FadeOut(star), 
                  FadeOut(div_label), FadeOut(cyclone), FadeOut(curl_label), 
                  FadeOut(physics_icon))
