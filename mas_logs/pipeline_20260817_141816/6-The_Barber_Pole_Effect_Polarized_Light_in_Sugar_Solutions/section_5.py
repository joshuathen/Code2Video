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
        self.setup_layout("Application: The Saccharimeter", [
            "Saccharimeters measure light rotation for purity.",
            "They detect sugar concentration in industrial syrup.",
            "This ensures consistent quality control."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Saccharimeters measure light rotation for purity.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Draw the saccharimeter setup
        source = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg", color=YELLOW)
        tube = Rectangle(height=0.5, width=3, color=WHITE, fill_opacity=0.3)
        polarizer = Rectangle(height=1.5, width=0.2, color=GRAY, fill_opacity=0.8)
        
        self.place_at_grid(source, 'B2', scale_factor=0.6)
        self.place_at_grid(polarizer, 'B3', scale_factor=0.6)
        self.place_in_area(tube, 'B4', 'B5', scale_factor=0.7)
        
        self.play(FadeIn(source), FadeIn(polarizer), Create(tube))
        
        # === Animation for Lecture Line 2 ===
        # They detect sugar concentration in industrial syrup.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Show sugar solution
        solution = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/syrup.svg")
        solution.set_fill("#E6E6FA", opacity=0.8)
        self.place_in_area(solution, 'D4', 'D5', scale_factor=0.6)
        self.play(FadeIn(solution))
        
        # === Animation for Lecture Line 3 ===
        # This ensures consistent quality control.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Adjust analyzer angle
        analyzer = Rectangle(height=1.5, width=0.2, color=GRAY, fill_opacity=0.8)
        self.place_at_grid(analyzer, 'D6', scale_factor=0.7)
        self.play(FadeIn(analyzer))
        self.play(Rotate(analyzer, angle=PI/6, about_point=self.grid['D6']))
        
        self.wait(2)
