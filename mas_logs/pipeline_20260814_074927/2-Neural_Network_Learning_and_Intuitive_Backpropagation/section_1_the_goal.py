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

class Section1TheGoalScene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "The Learning Goal: Pixel the Robot's Task"
        lines = [
            "Meet Pixel, a robot learning to see.",
            "Pixel's guesses are compared to the truth.",
            "The error, or loss, shows how wrong he is."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # A blurry dog image appears on the left (#FFFFFF). 
        # Pixel (#00FF00) appears on the right with a thought bubble showing a toaster (#C0C0C0).
        
        # Blurry dog image representation
        dog_box = Square(color=WHITE, fill_opacity=0.3)
        dog_label_internal = Text("DOG", font_size=36, color=WHITE).set_opacity(0.4)
        dog_image = VGroup(dog_box, dog_label_internal)
        self.place_in_area(dog_image, "B1", "D2", scale_factor=0.8)
        
        # Pixel the Robot
        pixel_body = Square(color="#00FF00", fill_opacity=1).scale(0.5)
        pixel_head = Square(color="#00FF00", fill_opacity=1).scale(0.25).next_to(pixel_body, UP, buff=0.05)
        pixel_eye_l = Dot(color=BLACK).move_to(pixel_head.get_left() + RIGHT*0.1 + UP*0.05).scale(0.6)
        pixel_eye_r = Dot(color=BLACK).move_to(pixel_head.get_right() + LEFT*0.1 + UP*0.05).scale(0.6)
        pixel_mouth_line = Line(pixel_head.get_bottom()+UP*0.1+LEFT*0.1, pixel_head.get_bottom()+UP*0.1+RIGHT*0.1, color=BLACK)
        pixel = VGroup(pixel_body, pixel_head, pixel_eye_l, pixel_eye_r, pixel_mouth_line)
        self.place_in_area(pixel, "C5", "D6", scale_factor=1.0)
        
        # Thought bubble
        bubble1 = Circle(radius=0.1, color="#C0C0C0", fill_opacity=0.8).move_to(pixel_head.get_corner(UR) + UP*0.2 + RIGHT*0.2)
        bubble2 = Circle(radius=0.15, color="#C0C0C0", fill_opacity=0.8).next_to(bubble1, UP+RIGHT, buff=0.1)
        thought_main = Ellipse(width=1.5, height=1.0, color="#C0C0C0", fill_opacity=0.2).next_to(bubble2, UP+RIGHT, buff=0.1)
        toaster_icon = Rectangle(width=0.8, height=0.5, color="#C0C0C0", fill_opacity=0.6).move_to(thought_main.get_center())
        thought_bubble = VGroup(bubble1, bubble2, thought_main, toaster_icon)
        
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(FadeIn(dog_image))
        self.play(Create(pixel))
        self.play(FadeIn(thought_bubble))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Text 'Toaster' (#FF0000) appears above Pixel. Text 'Dog' (#00FF00) appears above the image.
        
        toaster_text = Text("Toaster", color="#FF0000", font_size=32)
        self.place_at_grid(toaster_text, "A5")
        
        dog_text = Text("Dog", color="#00FF00", font_size=32)
        self.place_at_grid(dog_text, "A1")
        
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FF0000"))
        self.play(Write(dog_text), Write(toaster_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A bar graph labeled 'Loss' (#FFD700) rises to 80% height. Pixel looks sad (#FF0000).
        
        loss_label = Text("Loss", color="#FFD700", font_size=28)
        self.place_at_grid(loss_label, "F3")
        
        # Bar graph container
        bar_bg = Rectangle(width=0.8, height=3.0, color=GREY, fill_opacity=0.2)
        self.place_in_area(bar_bg, "B3", "E3") # Place in column 3
        
        # Bar fill
        bar_fill = Rectangle(width=0.8, height=0.01, color="#FFD700", fill_opacity=1)
        bar_fill.move_to(bar_bg.get_bottom(), aligned_edge=DOWN)
        
        # Sad face
        sad_mouth = Arc(radius=0.12, start_angle=0, angle=PI, color=BLACK).move_to(pixel_mouth_line.get_center() + DOWN*0.05)
        
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFD700"))
        self.play(FadeIn(loss_label), Create(bar_bg))
        self.play(
            bar_fill.animate.stretch_to_fit_height(3.0 * 0.8, about_edge=DOWN),
            pixel_mouth_line.animate.become(sad_mouth),
            pixel_body.animate.set_color("#FF0000"),
            pixel_head.animate.set_color("#FF0000")
        )
        self.wait(2)
