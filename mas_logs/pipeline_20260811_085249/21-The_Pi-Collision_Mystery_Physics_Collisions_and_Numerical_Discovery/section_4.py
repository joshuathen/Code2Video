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
        self.setup_layout("Application and Conclusion", [
            "Collisions perform complex mathematical calculations.",
            "Mechanical rules effectively act as a computer.",
            "We witness transcendental constants in physical motion."
        ])
        
        # Assets
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=BLUE)
        particles = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particles.svg", color=YELLOW)
        
        circ = Circle(radius=1.0, color=WHITE).set_stroke(width=2)
        
        combined_visual_group = VGroup(computer, circ, particles)
        self.place_in_area(combined_visual_group, 'B2', 'D5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Show the full simulation path in the geometry using computer.svg
        self.play(DrawBorderThenFill(computer), Create(circ), run_time=1.5)
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Animate particles emerging from the collision region.
        # Fixed layout: self.place_at_grid(circ, 'B3', scale_factor=0.6)
        self.place_at_grid(circ, 'B3', scale_factor=0.6)
        self.play(FadeIn(particles.scale(0.5).next_to(circ, RIGHT)), run_time=1.5)
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        # Summarize the connection between collisions and Pi using particles.svg
        label = Text("3.1415...", font_size=24, color=GOLD)
        self.place_at_grid(label, 'C3', scale_factor=0.7)
        self.play(Write(label), particles.animate.set_color(GOLD), run_time=1.5)
        self.lecture[2].set_color(GOLD)
        
        self.wait(2)
