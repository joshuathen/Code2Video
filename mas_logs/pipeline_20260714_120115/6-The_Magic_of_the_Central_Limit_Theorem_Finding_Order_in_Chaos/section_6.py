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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the specific title and lecture lines from the storyboard
        self.setup_layout(
            "Application: Why It Matters in the Real World",
            [
                "The CLT allows predictions without knowing the population shape.",
                "It provides confidence for polling and scientific research.",
                "Order emerges from chaos through the power of averages."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)

        # Create Axes for the distribution comparison in rows B-C
        # This illustrates the ability to predict without knowing the original shape
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1, 0.5],
            axis_config={"include_tip": False, "color": GREY},
            x_length=3.5,
            y_length=1.5
        )
        self.place_in_area(axes, "B2", "C6")
        
        # Jagged/Chaos distribution representation (#FFFFFF)
        jagged_data = [
            (0, 0.2), (1, 0.8), (2, 0.1), (3, 0.6), (4, 0.3),
            (5, 0.9), (6, 0.2), (7, 0.7), (8, 0.4), (9, 0.6), (10, 0.1)
        ]
        jagged_curve = axes.plot_line_graph(
            x_values=[x for x, y in jagged_data],
            y_values=[y for x, y in jagged_data],
            add_vertex_dots=False,
            line_color=WHITE
        )
        
        # Smooth Normal distribution (The Order)
        # Using a scaled Gaussian function to fit the axes
        smooth_curve = axes.plot(
            lambda x: np.exp(-0.5 * ((x - 5) / 1.5)**2) / (1.5 * np.sqrt(2 * np.pi)) * 3,
            color=WHITE,
            x_range=[0, 10]
        )
        
        self.play(Create(axes), Create(jagged_curve))
        self.wait(0.5)
        # Morphing jagged distribution into a smooth curve
        self.play(ReplacementTransform(jagged_curve, smooth_curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlighting
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Display icons for real-world applications
        # Opinion Polls: Ballot icon from [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/ballot.svg]
        ballot_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/ballot.svg")
        ballot_icon.set_color(WHITE)
        self.place_at_grid(ballot_icon, "D3", scale_factor=0.6)
        ballot_label = Text("Opinion Polls", font_size=16).next_to(ballot_icon, RIGHT, buff=0.2)
        
        # Scientific Research: Medical Cross icon (#87CEFA)
        medical_cross = VGroup(
            Rectangle(width=0.4, height=0.1, fill_opacity=1, stroke_width=0),
            Rectangle(width=0.1, height=0.4, fill_opacity=1, stroke_width=0)
        ).set_color("#87CEFA")
        self.place_at_grid(medical_cross, "D5", scale_factor=1.0)
        medical_label = Text("Medical Trials", font_size=16, color="#87CEFA").next_to(medical_cross, RIGHT, buff=0.2)

        self.play(
            FadeIn(ballot_icon), Write(ballot_label),
            FadeIn(medical_cross), Write(medical_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final lecture highlighting
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Final summary text (#FFFF00)
        final_summary = Text("Order from Chaos", color=YELLOW, font_size=32)
        # Positioned in Row E to avoid the bottom boundary (Row F) and provide balance
        self.place_in_area(final_summary, "E2", "E6", scale_factor=0.9)
        
        self.play(FadeIn(final_summary, shift=UP * 0.2))
        self.wait(3)
