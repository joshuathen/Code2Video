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
        self.setup_layout("Summary and Synthesis", [
            "Information theory creates computational efficiency.",
            "Minimize bits of remaining uncertainty.",
            "Collapse candidates into the target."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Reiterate the entropy concept summary
        entropy_text = Text("Entropy = Uncertainty", font_size=36, color="#FFFFFF")
        self.place_at_grid(entropy_text, "B2", scale_factor=0.9)
        self.play(FadeIn(entropy_text))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the optimal strategy flowchart
        flowchart = VGroup(
            Rectangle(height=0.5, width=1.5, color=GREY),
            Arrow(UP, DOWN, color=GREY),
            Rectangle(height=0.5, width=1.5, color=GREY)
        ).arrange(DOWN)
        self.place_in_area(flowchart, "A3", "C5", scale_factor=0.8)
        
        self.play(Create(flowchart))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final takeaway: uncertainty reduction is key
        dot_cloud = VGroup(*[Dot(radius=0.03) for _ in range(50)])
        dot_cloud.arrange_in_grid(5, 10, buff=0.1)
        self.place_at_grid(dot_cloud, "E4", scale_factor=0.9)
        
        self.play(FadeIn(dot_cloud))
        self.play(dot_cloud.animate.scale(0.1).set_opacity(0.3))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
