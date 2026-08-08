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
        self.setup_layout(
            "Summary and Real-World Application",
            [
                "Interference records phase, while diffraction reconstructs the scene.",
                "This technology secures credit cards with shimmering holograms.",
                "Holography also enables advanced medical and data storage systems."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Interference records phase, while diffraction reconstructs the scene.
        # Flowchart: 'Interference' (Record) #00FFFF -> 'Diffraction' (Reconstruct) #00FF00.
        
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        interference_text = Text("Interference\n(Record)", font_size=20, color="#00FFFF")
        diffraction_text = Text("Diffraction\n(Reconstruct)", font_size=20, color="#00FF00")
        
        self.place_at_grid(interference_text, "B3")
        self.place_at_grid(diffraction_text, "D3")
        
        flow_arrow = Arrow(
            start=interference_text.get_bottom(),
            end=diffraction_text.get_top(),
            buff=0.1,
            color=WHITE
        )
        
        flow_group = VGroup(interference_text, flow_arrow, diffraction_text)
        self.play(FadeIn(flow_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # This technology secures credit cards with shimmering holograms.
        # Display a credit card #FFFFFF. Tilt it to show a shimmering holographic bird changing color and shape.
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFFF"),
            FadeOut(flow_group)
        )
        
        # Credit Card
        card = RoundedRectangle(corner_radius=0.1, height=1.5, width=2.5, color=WHITE, fill_opacity=0.1)
        # B001/Issue 42: place in area B2 to E5
        self.place_in_area(card, 'B2', 'E5')
        
        # Bird (represented by a Star that will change shape/color)
        bird = Star(n=5, color="#00FFFF", fill_opacity=0.8).scale(0.3)
        bird.move_to(card.get_center())
        
        # Group them for simultaneous rotation
        card_group = VGroup(card, bird)
        
        self.play(FadeIn(card_group))
        
        # Shimmering effect: color and shape changes with card tilting
        self.play(
            Rotate(card_group, angle=20*DEGREES, axis=RIGHT),
            Rotate(card_group, angle=20*DEGREES, axis=UP),
            bird.animate.set_color("#FF00FF").scale(1.2),
            run_time=1.5
        )
        self.play(
            Rotate(card_group, angle=-40*DEGREES, axis=RIGHT),
            Rotate(card_group, angle=-40*DEGREES, axis=UP),
            bird.animate.set_color("#FFFF00").scale(0.8),
            run_time=1.5
        )
        self.play(
            Rotate(card_group, angle=20*DEGREES, axis=RIGHT),
            Rotate(card_group, angle=20*DEGREES, axis=UP),
            bird.animate.set_color("#00FFFF").scale(1.0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Holography also enables advanced medical and data storage systems.
        # Icons for 'Data Storage' and 'Medical Imaging' appear #FFFF00 to show modern utility.
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            FadeOut(card_group)
        )
        
        # Data Storage Icon (Simplified: Square with lines)
        storage_icon = VGroup(
            Square(side_length=0.8, color="#FFFF00"),
            Line(LEFT, RIGHT, color="#FFFF00").scale(0.3).shift(UP*0.2),
            Line(LEFT, RIGHT, color="#FFFF00").scale(0.3),
            Line(LEFT, RIGHT, color="#FFFF00").scale(0.3).shift(DOWN*0.2)
        )
        storage_label = Text("Data Storage", font_size=18, color="#FFFF00")
        storage_group = VGroup(storage_icon, storage_label).arrange(RIGHT, buff=0.5)
        
        # Medical Imaging Icon (Simplified: Circle with a cross)
        medical_icon = VGroup(
            Circle(radius=0.4, color="#FFFF00"),
            Line(UP, DOWN, color="#FFFF00").scale(0.3),
            Line(LEFT, RIGHT, color="#FFFF00").scale(0.3)
        )
        medical_label = Text("Medical Imaging", font_size=18, color="#FFFF00")
        medical_group = VGroup(medical_icon, medical_label).arrange(RIGHT, buff=0.5)
        
        # Positioning according to Issues 43 and 44
        # Issue 43: storage_group in B2 to B6
        self.place_in_area(storage_group, 'B2', 'B6')
        
        # Issue 44: medical_group in E2 to E6
        self.place_in_area(medical_group, 'E2', 'E6')
        
        self.play(
            FadeIn(storage_group),
            FadeIn(medical_group)
        )
        self.wait(3)
